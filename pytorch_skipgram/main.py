from pytorch_skipgram.model import SkipGram
from pytorch_skipgram.utils.vocab import Corpus

from torch import optim
from torch import tensor
import torch

import numpy as np
import argparse

rnd = np.random.RandomState(7)

parser = argparse.ArgumentParser(description='Skip-gram with Negative Sampling by PyTorch')
parser.add_argument('--window', type=int, default=5, metavar='ws',
                    help='the number of window size (default: 5)')
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
parser.add_argument('--epoch', type=int, default=7, metavar='epochs',
                    help='the number of epochs (default: 7)')
parser.add_argument('--lr', type=float, default=0.025, metavar='starting_lr',
                    help='initial learning rate (default: 0.025)')
parser.add_argument('--input', type=str, metavar='fname',
                    help='training corpus file name')
parser.add_argument('--out', type=str, metavar='outfname',
                    help='vector file name')


args = parser.parse_args()

starting_lr = args.lr
ws = args.window

print("Load corpus")
corpus = Corpus(min_count=args.min_count)
docs = corpus.tokenize_from_file(args.input)
corpus.build_discard_table(t=args.samples)
print("V:{}, #words:{}".format(corpus.num_vocab, corpus.num_words))

model = SkipGram(corpus=corpus, noise_param=args.noise, num_negatives=args.negative, embedding_dim=args.dim)


def train(epochs, rnd=np.random.RandomState(7)):
    def update_lr(starting_lr, num_processed_words, epochs, num_words):
        new_lr = starting_lr * (1. - num_processed_words / (epochs * num_words + 1))
        lower_lr = starting_lr * 0.0001
        return max(new_lr, lower_lr)

    def generate_words_from_doc(doc, num_processed_words):
        """
        generator aims to separate too long document into some shorter documents
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

    optimizer = optim.SGD(model.parameters(), lr=starting_lr)
    model.train()
    num_processed_words = 0
    num_words = corpus.num_words
    last_check = 0

    for epoch in range(epochs):
        for doc in docs:
            for doc, num_processed_words in generate_words_from_doc(doc=doc, num_processed_words=num_processed_words):
                # update lr and logging
                if num_processed_words - last_check > 10000:
                    optimizer.param_groups[0]['lr'] = lr = update_lr(starting_lr, num_processed_words, epochs, num_words)
                    print('\rprogress: {0:.5f}, lr={1:.5f}'.format(
                        num_processed_words / (num_words * epochs), lr),
                        end='')
                    last_check = num_processed_words

                doclen = len(doc)
                dynamic_window_sizes = rnd.randint(low=1, high=ws+1, size=doclen)
                for (position, (word_id, dynamic_window_size)) in enumerate(zip(doc, dynamic_window_sizes)):
                    begin_pos = max(0, position-dynamic_window_size)
                    end_pos = min(position+dynamic_window_size, doclen-1) + 1
                    targets = []
                    for context_position in range(begin_pos, end_pos):
                        if context_position == position:
                            continue
                        targets.append([doc[context_position]])

                    targets = tensor(torch.LongTensor(targets), requires_grad=False)
                    inputs = tensor(torch.LongTensor([[word_id]]), requires_grad=False)
                    optimizer.zero_grad()
                    loss = model(inputs, targets)
                    loss.backward()
                    optimizer.step()


train(epochs=args.epoch, rnd=rnd)

with open(args.out, 'w') as f:
    f.write('{} {}\n'.format(corpus.num_vocab, args.dim))
    for word_id, vec in enumerate(model.in_embeddings.weight.data.numpy()):
        word = corpus.dictionary.id2word[word_id]
        vec = ' '.join(list(map(str, vec)))
        f.write('{} {}\n'.format(word, vec))
