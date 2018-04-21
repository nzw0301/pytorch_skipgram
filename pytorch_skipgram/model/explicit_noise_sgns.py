import torch
import torch.nn as nn


class EXPSkipGram(nn.Module):
    def __init__(self, V, embedding_dim=100):
        super(EXPSkipGram, self).__init__()

        self.embedding_dim = embedding_dim
        self.in_embeddings = nn.Embedding(V, embedding_dim, sparse=True)
        self.out_embeddings = nn.Embedding(V, embedding_dim, sparse=True)
        self.log_sigmoid = nn.LogSigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        upper = 0.5/self.embedding_dim
        self.in_embeddings.weight.data.uniform_(-upper, upper)
        self.out_embeddings.weight.data.zero_()

    def forward(self, inputs, contexts, negatives):
        """
        :param inputs: (#mini_batches, 1)
        :param contexts: (#mini_batches, 1)
        :param negatives: (#mini_batches, #negatives)
        :return:
        """
        in_vectors = self.in_embeddings(inputs)
        context_pos_vectors = self.out_embeddings(contexts)
        context_neg_vectors = self.out_embeddings(negatives)

        pos = torch.sum(torch.mul(in_vectors, context_pos_vectors), -1)
        neg = torch.sum(torch.mul(in_vectors, context_neg_vectors), -1)
        return - torch.mean(self.log_sigmoid(pos).view(-1) + torch.sum(self.log_sigmoid(-neg), -1))
