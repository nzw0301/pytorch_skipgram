import logging

import hydra
import numpy as np
import torch
from torch import optim

from pytorch_skipgram.loss import negative_sampling_loss, nce_loss
from pytorch_skipgram.model import SkipGram
from pytorch_skipgram.utils.negative_sampler import NegativeSampler
from pytorch_skipgram.utils.vocab import Corpus

NEGATIVE_TABLE_SIZE = int(1e8)


def update_lr(starting_lr, num_processed_words, epochs, num_words):
    new_lr = starting_lr * (1. - num_processed_words / (epochs * num_words + 1))
    lower_lr = starting_lr * 0.0001
    return max(new_lr, lower_lr)


def generate_words_from_doc(doc, num_processed_words, corpus, rnd):
    """
    this generator separates a long document into shorter documents
    :param doc: np.array(np.int), word_id list
    :param num_processed_words:
    :return: shorter words list: np.array(np.int), incremented num processed words: num_processed_words
    """
    new_doc = []
    for word_id in doc:
        num_processed_words += 1
        if corpus.discard(word_id=word_id, rnd=rnd):
            continue
        new_doc.append(word_id)
        if len(new_doc) >= 1000:
            yield np.array(new_doc), num_processed_words
            new_doc = []
    yield np.array(new_doc), num_processed_words


def train_on_minibatches(model, optimizer, use_cuda, inputs, contexts, negatives, is_neg_loss, log_k_prob=None):
    num_minibatches = len(contexts)
    inputs = torch.LongTensor(inputs).view(num_minibatches, 1)
    if use_cuda:
        inputs = inputs.cuda()

    optimizer.zero_grad()

    if is_neg_loss:
        contexts = torch.LongTensor(contexts).view(num_minibatches, 1)
        negatives = torch.LongTensor(negatives)
        if use_cuda:
            contexts = contexts.cuda()
            negatives = negatives.cuda()

        pos, neg = model.forward(inputs, contexts, negatives)
        loss = negative_sampling_loss(pos, neg)
    else:
        pos_log_k_negative_prob = torch.FloatTensor(log_k_prob[contexts]).view(num_minibatches, 1)
        neg_log_k_negative_prob = torch.FloatTensor(log_k_prob[negatives])

        contexts = torch.LongTensor(contexts).view(num_minibatches, 1)
        negatives = torch.LongTensor(negatives)
        if use_cuda:
            pos_log_k_negative_prob = pos_log_k_negative_prob.cuda()
            neg_log_k_negative_prob = neg_log_k_negative_prob.cuda()
            contexts = contexts.cuda()
            negatives = negatives.cuda()

        pos, neg = model.forward(inputs, contexts, negatives)
        loss = nce_loss(pos, neg, pos_log_k_negative_prob, neg_log_k_negative_prob)

    loss.backward()
    optimizer.step()
    return loss.item()


@hydra.main(config_path='../conf/config.yaml')
def main(cfg):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ''
    logger.addHandler(stream_handler)

    ws = cfg['parameters']['window']
    num_negatives = cfg['parameters']['negative']
    epochs = cfg['parameters']['epochs']
    num_minibatches = cfg['parameters']['batch']
    starting_lr = cfg['parameters']['lr'] * num_minibatches
    lr_update_rate = cfg['parameters']['lr_update_rate']
    embedding_dim = cfg['parameters']['dim']

    seed = cfg['experiments']['seed']
    rnd = np.random.RandomState(seed)
    torch.manual_seed(seed)

    logger.info('Loading training corpus...\n')
    corpus = Corpus(min_count=cfg['parameters']['min_count'])
    docs = corpus.tokenize_from_file(cfg['dataset']['input_path'])
    corpus.build_discard_table(t=cfg['parameters']['samples'])
    logger.info('V:{}, #words:{}\n'.format(corpus.num_vocab, corpus.num_words))

    is_neg_loss = (cfg['parameters']['loss'] == 'neg')
    negative_sampler = NegativeSampler(
        frequency=corpus.dictionary.id2freq,
        negative_alpha=cfg['parameters']['noise'],
        is_neg_loss=is_neg_loss,
        table_length=NEGATIVE_TABLE_SIZE
    )
    if is_neg_loss:
        log_k_prob = None
        logger.info('loss function: Negative Sampling\n')
    else:
        log_k_prob = np.log(num_negatives * negative_sampler.noise_dist)
        logger.info('loss function: NCE\n')

    model = SkipGram(V=corpus.num_vocab, embedding_dim=embedding_dim)

    use_cuda = torch.cuda.is_available() and cfg['experiments']['gpu_id'] > -1
    if use_cuda:
        torch.cuda.set_device(cfg['experiments']['gpu_id'])
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=starting_lr)
    model.train()
    num_processed_words = last_check = 0
    num_words = corpus.num_words
    loss_value = 0
    num_add_loss_value = 0
    for epoch in range(epochs):
        inputs = []
        contexts = []
        for sentence in docs:
            for doc, num_processed_words in generate_words_from_doc(
                    doc=sentence, num_processed_words=num_processed_words, corpus=corpus, rnd=rnd
            ):

                doclen = len(doc)
                dynamic_window_sizes = rnd.randint(low=1, high=ws + 1, size=doclen)
                for (position, (word_id, dynamic_window_size)) in enumerate(zip(doc, dynamic_window_sizes)):
                    begin_pos = max(0, position - dynamic_window_size)
                    end_pos = min(position + dynamic_window_size, doclen - 1) + 1
                    for context_position in range(begin_pos, end_pos):
                        if context_position == position:
                            continue
                        contexts.append(doc[context_position])
                        inputs.append(word_id)
                        if len(inputs) >= num_minibatches:
                            negatives = negative_sampler.sample(k=num_negatives, rnd=rnd, exclude_words=contexts)
                            loss_value += train_on_minibatches(
                                model=model,
                                optimizer=optimizer,
                                use_cuda=use_cuda,
                                inputs=inputs,
                                contexts=contexts,
                                negatives=negatives,
                                is_neg_loss=is_neg_loss,
                                log_k_prob=log_k_prob
                            )
                            num_add_loss_value += 1
                            inputs.clear()
                            contexts.clear()
                if len(inputs) > 0:
                    negatives = negative_sampler.sample(k=num_negatives, rnd=rnd, exclude_words=contexts)
                    loss_value += train_on_minibatches(
                        model=model,
                        optimizer=optimizer,
                        use_cuda=use_cuda,
                        inputs=inputs,
                        contexts=contexts,
                        negatives=negatives,
                        is_neg_loss=is_neg_loss,
                        log_k_prob=log_k_prob
                    )
                    num_add_loss_value += 1
                    inputs.clear()
                    contexts.clear()

                # update lr and print progress
                if num_processed_words - last_check > lr_update_rate:
                    optimizer.param_groups[0]['lr'] = lr = update_lr(starting_lr,
                                                                     num_processed_words,
                                                                     epochs,
                                                                     num_words)

                    logger.info('\rprogress: {0:.7f}, lr={1:.7f}, loss={2:.7f}'.format(
                        num_processed_words / (num_words * epochs),
                        lr, loss_value / num_add_loss_value),
                    )
                    last_check = num_processed_words

    with open(cfg['dataset']['outout_file_name'], 'w') as f:
        f.write('{} {}\n'.format(corpus.num_vocab, embedding_dim))
        if use_cuda:
            embeddings = model.in_embeddings.weight.data.cpu().numpy()
        else:
            embeddings = model.in_embeddings.weight.data.numpy()
        for word_id, vec in enumerate(embeddings):
            word = corpus.dictionary.id2word[word_id]
            vec = ' '.join(list(map(str, vec)))
            f.write('{} {}\n'.format(word, vec))


if __name__ == '__main__':
    main()
