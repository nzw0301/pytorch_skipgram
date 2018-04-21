from pytorch_skipgram.utils import Corpus
import numpy as np
import os
import string

text8_like_file_name = 'text8-like.txt'
doc_file_name = 'doc.txt'

def create_embeddings_files():
    with open(text8_like_file_name, 'w') as f:
        words = []
        for count, char in enumerate(string.ascii_lowercase, start=1):
            for _ in range(count):
                words.append(char)
        f.write(' '.join(words))

    with open(doc_file_name, 'w') as f:
        for count, char in enumerate(string.ascii_lowercase, start=1):
            words = []
            for _ in range(count):
                words.append(char)
            f.write(' '.join(words) + '\n')


def delete_embeddings_files():
    os.remove(text8_like_file_name)
    os.remove(doc_file_name)


def test_corpus():
    create_embeddings_files()

    def test_tokenize_from_file(fname):
        corpus = Corpus(min_count=5)
        corpus.tokenize_from_file(fname)
        assert len(corpus.dictionary) == 26-4
        assert corpus.dictionary.id2word[0] == 'z'
        assert corpus.dictionary.id2word[-1] == 'e'
        assert corpus.dictionary.id2freq[0] == 26
        assert corpus.num_vocab == 26 - 4
        assert corpus.num_words == np.sum(np.arange(5, 27))

        if 'text8' in fname:
            assert corpus.num_docs == 1
        else:
            assert corpus.num_docs == 26 - 4

    test_tokenize_from_file(text8_like_file_name)
    test_tokenize_from_file(doc_file_name)
    delete_embeddings_files()
