import numpy as np
import os
import json
import collections
import linecache

from net.multimodal.data_provider.vocab_data import Vocabulary


class Word2VecData(object):

    def __init__(self, w2v_vocab_fname, w2v_vectors_fname):

        self.w2v_vocab_fname = w2v_vocab_fname
        self.w2v_vectors_fname = w2v_vectors_fname

        self.word2vec_dim = 0
        self.word2vec_vocab = None

        return

    def set_word2vec_vocab(self):
        self.word2vec_vocab = Vocabulary(self.w2v_vocab_fname)

        # with open(self.d['word2vec_vocab'], 'rb') as f:
        #     self.word2vec_vocab = [w.replace('\n', '') for w in f.readlines()]

    def set_word2vec_dim(self):
        # read the first line of word2vec vector file and check the dimension
        vec = np.fromstring(linecache.getline(self.w2v_vectors_fname, 1), sep=" ")
        self.word2vec_dim = vec.shape[0]

    def get_word2vec_dim(self):
        if self.word2vec_dim == 0:
            self.set_word2vec_dim()
        return self.word2vec_dim

    def get_word_vectors_of_word_list(self, word_list):

        if self.word2vec_vocab is None:
            self.set_word2vec_vocab()

        if self.word2vec_dim == 0:
            self.set_word2vec_dim()

        word2id, id2word = self.word2vec_vocab.get_vocab_dicts()

        X_txt = np.zeros((len(word_list), self.word2vec_dim))

        i = 0
        for word in word_list:
            if word not in word2id:
                raise ValueError("word should be in w2v vocab by constructiton. Something went wrong")
            w_id = word2id[word]
            # X_txt[i, :] = self.word2vec_vectors[w_id, :]
            # Note that linecache line numbers start at 1.
            X_txt[i, :] = np.fromstring(linecache.getline(self.w2v_vectors_fname, w_id + 1), sep=" ")
            i += 1

        return X_txt
