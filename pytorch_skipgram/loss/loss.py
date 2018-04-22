import torch
from torch.nn.functional import logsigmoid


def nce_loss(pos_dot, neg_dot, pos_log_k_negative_prob, neg_log_k_negative_prob, size_average=True, reduce=True):
    """
    https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf

    :param pos_dot:
    :param neg_dot:
    :param pos_log_k_negative_prob:
    :param neg_log_k_negative_prob:
    :param size_average:
    :param reduce:
    :return:
    """
    s_pos = pos_dot - pos_log_k_negative_prob
    s_neg = neg_dot - neg_log_k_negative_prob
    loss = - (torch.mean(logsigmoid(s_pos).view(-1) + torch.sum(logsigmoid(-s_neg), -1)))

    if not reduce:
        return loss
    if size_average:
        return torch.mean(loss)
    return torch.sum(loss)


def negative_sampling_loss(pos_dot, neg_dot, size_average=True, reduce=True):
    """
    :param pos_dot:
    :param neg_dot:
    :param size_average:
    :param reduce:
    :return:
    """
    loss = -(logsigmoid(pos_dot).view(-1) + torch.sum(logsigmoid(-neg_dot), -1))
    if not reduce:
        return loss
    if size_average:
        return torch.mean(loss)
    return torch.sum(loss)