import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word2id = {}
        self.id2word = []
        self.word2freq = {}
        self.id2freq = None

    def add_word(self, word):
        if word not in self.word2id:
            self.id2word.append(word)
            self.word2id[word] = len(self.id2word) - 1
            self.word2freq[word] = 1
        else:
            self.word2freq[word] += 1

    def rebuild(self, min_count=5):
        self.id2word = sorted(self.word2freq, key=self.word2freq.get, reverse=True)

        for new_word_id, word in enumerate(self.id2word):
            freq = self.word2freq[word]
            if freq >= min_count:
                self.word2id[word] = new_word_id
            else:
                for word in self.id2word[new_word_id:]:
                    del self.word2id[word]
                self.id2word = self.id2word[:new_word_id]
                self.id2freq = np.array([self.word2freq[word] for word in self.id2word])
                del self.word2freq
                break

    def __len__(self):
        return len(self.id2word)


class Corpus(object):
    def __init__(self, min_count=5):
        self.dictionary = Dictionary()
        self.min_count = min_count
        self.num_words = 0
        self.num_vocab = 0
        self.num_docs = 0
        self.discard_table = None

    def tokenize_from_file(self, path):
        self.num_words = 0
        self.num_docs = 0
        with open(path) as f:
            for l in f:
                for word in l.strip().split():
                    self.dictionary.add_word(word=word)
        self.dictionary.rebuild(min_count=self.min_count)
        self.num_vocab = len(self.dictionary)

        with open(path) as f:
            docs = []
            for l in f:
                doc = []
                for word in l.strip().split():
                    if word in self.dictionary.word2id:
                        doc.append(self.dictionary.word2id.get(word))
                if len(doc) > 1:
                    docs.append(doc)
                    self.num_words += len(doc)
                    self.num_docs += 1

        return np.array(docs)

    def build_discard_table(self, t=1e-4):
        # https://github.com/facebookresearch/fastText/blob/53dd4c5cefec39f4cc9f988f9f39ab55eec6a02f/src/dictionary.cc#L277
        tf = t / (self.dictionary.id2freq / self.num_words)
        self.discard_table = np.sqrt(tf) + tf

    def discard(self, word_id, rnd):
        return rnd.rand() > self.discard_table[word_id]
