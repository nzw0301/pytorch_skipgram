from torch import optim
import torch

import numpy as np
import argparse
import logging

from pytorch_skipgram.model import SkipGram
from pytorch_skipgram.loss import negative_sampling_loss, nce_loss
from pytorch_skipgram.utils.vocab import Corpus
from pytorch_skipgram.utils.negative_sampler import NegativeSampler


NEGATIVE_TABLE_SIZE = int(1e8)


def update_lr(starting_lr, num_processed_words, epochs, num_words):
    new_lr = starting_lr * (1. - num_processed_words / (epochs * num_words + 1))
    lower_lr = starting_lr * 0.0001
    return max(new_lr, lower_lr)


def generate_words_from_doc(doc, num_processed_words):
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


def train_on_minibatches(model, optimizer, use_cuda, inputs, contexts, negatives):
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


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ''
    logger.addHandler(stream_handler)

    parser = argparse.ArgumentParser(description='Skip-gram with Negative Sampling by PyTorch')
    parser.add_argument('--window', type=int, default=5, metavar='ws',
                        help='the number of windows (default: 5)')
    parser.add_argument('--dim', type=int, default=100, metavar='dim',
                        help='the number of vector dimensions (default: 100)')
    parser.add_argument('--min-count', type=int, default=5, metavar='min',
                        help='threshold value for lower frequency words (default: 5)')
    parser.add_argument('--samples', type=float, default=1e-3, metavar='t',
                        help='sub-sampling parameter (default: 1e-3)')
    parser.add_argument('--noise', type=float, default=0.75, metavar='noise',
                        help='power value of noise distribution (default: 0.75)')
    parser.add_argument('--negative', type=int, default=5, metavar='neg',
                        help='the number of negative samples (default: 5)')
    parser.add_argument('--epochs', type=int, default=7, metavar='epochs',
                        help='the number of epochs (default: 7)')
    parser.add_argument('--batch', type=int, default=512, metavar='num_minibatches',
                        help='the number of pairs of words (default: 512)')
    parser.add_argument('--lr_update_rate', type=int, default=1000,
                        help='update scheduler lr (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.025, metavar='starting_lr',
                        help='initial learning rate (default: 0.025) internal learning rate is `lr * num_minibatch`')
    parser.add_argument('--input', type=str, metavar='fname',
                        help='training corpus file name')
    parser.add_argument('--out', type=str, metavar='outfname',
                        help='vector file name')
    parser.add_argument('--loss', type=str, default='neg',
                        help='loss function name: neg (negative sampling) or nce (noise contrastive estimation) (default: neg)')
    parser.add_argument('--gpu-id', type=int, default=-1, metavar='gpuid',
                        help='gpu id (default: -1, aka CPU)')
    parser.add_argument('--seed', type=int, default=7,
                        help='random seed value for numpy and pytorch. (default: 7)')

    args = parser.parse_args()
    ws = args.window
    num_negatives = args.negative
    epochs = args.epochs
    num_minibatches = args.batch
    starting_lr = args.lr * num_minibatches
    lr_update_rate = args.lr_update_rate
    rnd = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    logger.info('Loading training corpus...\n')
    corpus = Corpus(min_count=args.min_count)
    docs = corpus.tokenize_from_file(args.input)
    corpus.build_discard_table(t=args.samples)
    logger.info('V:{}, #words:{}\n'.format(corpus.num_vocab, corpus.num_words))

    is_neg_loss = (args.loss == 'neg')
    negative_sampler = NegativeSampler(
        frequency=corpus.dictionary.id2freq,
        negative_alpha=args.noise,
        is_neg_loss=is_neg_loss,
        table_length=NEGATIVE_TABLE_SIZE
    )
    if is_neg_loss:
        logger.info('loss function: Negative Sampling\n')
    else:
        log_k_prob = np.log(num_negatives * negative_sampler.noise_dist)
        logger.info('loss function: NCE\n')

    model = SkipGram(V=corpus.num_vocab, embedding_dim=args.dim)

    use_cuda = torch.cuda.is_available() and args.gpu_id > -1
    if use_cuda:
        torch.cuda.set_device(args.gpu_id)
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
            for doc, num_processed_words in generate_words_from_doc(doc=sentence,
                                                                    num_processed_words=num_processed_words):

                doclen = len(doc)
                dynamic_window_sizes = rnd.randint(low=1, high=ws+1, size=doclen)
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
                                negatives=negatives
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
                        negatives=negatives
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

    with open(args.out, 'w') as f:
        f.write('{} {}\n'.format(corpus.num_vocab, args.dim))
        if use_cuda:
            embeddings = model.in_embeddings.weight.data.cpu().numpy()
        else:
            embeddings = model.in_embeddings.weight.data.numpy()
        for word_id, vec in enumerate(embeddings):
            word = corpus.dictionary.id2word[word_id]
            vec = ' '.join(list(map(str, vec)))
            f.write('{} {}\n'.format(word, vec))