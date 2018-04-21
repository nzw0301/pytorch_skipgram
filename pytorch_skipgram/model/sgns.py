from pytorch_skipgram.utils import Corpus

from torch import tensor
import torch
import torch.nn as nn
import numpy as np


class NEG(nn.Module):
    def __init__(self, corpus, noise_param=0.75, num_negatives=10, rnd=np.random.RandomState(7), embedding_dim=100):
        super(NEG, self).__init__()
        self.NEGATIVE_TABLE_SIZE = 10_000_000
        self.num_negatives = num_negatives
        self.rnd = rnd
        self.negative_table = self.init_negative_table(corpus.dictionary.id2freq, noise_param)
        self.log_sigmoid = nn.LogSigmoid()
        self.out_embeddings = nn.Embedding(corpus.num_vocab, embedding_dim, sparse=True)
        self.out_embeddings.weight.data.zero_()

    def init_negative_table(self, frequency: np.ndarray, negative_alpha):
        z = np.sum(np.power(frequency, negative_alpha))
        negative_table = np.zeros(self.NEGATIVE_TABLE_SIZE, dtype=np.int32)
        begin_index = 0
        for word_id, freq in enumerate(frequency):
            c = np.power(freq, negative_alpha)
            end_index = begin_index + int(c * self.NEGATIVE_TABLE_SIZE / z) + 1
            negative_table[begin_index:end_index] = word_id
            begin_index = end_index
        return negative_table

    def sample_negatives(self, contexts):
        return tensor(torch.LongTensor(
            self.negative_table[self.rnd.randint(low=0, high=self.NEGATIVE_TABLE_SIZE, size=(contexts.shape[0], self.num_negatives))]
        ), requires_grad=False)

    def forward(self, in_vectors, contexts):
        context_pos_vectors = self.out_embeddings(contexts)
        context_neg_vectors = self.out_embeddings(self.sample_negatives(contexts))

        loss = torch.sum(self.log_sigmoid( torch.sum(in_vectors * context_pos_vectors, -1))) + \
               torch.sum(self.log_sigmoid(-torch.sum(in_vectors * context_neg_vectors, -1)))

        return -loss / contexts.shape[0]


class SkipGram(nn.Module):
    def __init__(self, corpus: Corpus, embedding_dim=100, noise_param=0.75, num_negatives=10, rnd=np.random.RandomState(7)):
        super(SkipGram, self).__init__()

        self.embedding_dim = embedding_dim
        self.in_embeddings = nn.Embedding(corpus.num_vocab, embedding_dim, sparse=True)
        self.neg = NEG(corpus, noise_param, num_negatives, rnd, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        upper = 0.5/self.embedding_dim
        self.in_embeddings.weight.data.uniform_(-upper, upper)

    def forward(self, inputs, contexts):
        in_vectors = self.in_embeddings(inputs)
        return self.neg(in_vectors, contexts)
