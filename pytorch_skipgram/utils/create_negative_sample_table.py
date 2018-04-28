import numpy as np


def init_negative_table(frequency: np.ndarray, negative_alpha, is_neg_loss, table_length):
    z = np.sum(np.power(frequency, negative_alpha))
    negative_table = np.zeros(table_length, dtype=np.int32)
    begin_index = 0
    for word_id, freq in enumerate(frequency):
        c = np.power(freq, negative_alpha)
        end_index = begin_index + int(c * table_length / z) + 1
        negative_table[begin_index:end_index] = word_id
        begin_index = end_index
    if is_neg_loss:
        return negative_table
    else:
        return negative_table, np.power(frequency, negative_alpha)
