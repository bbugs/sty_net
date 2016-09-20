

class Vocabulary(object):

    def __init__(self, fname):
        self.fname = fname
        self.vocab = []
        return

    def set_vocab(self):
        with open(self.fname, 'rb') as f:
            self.vocab = [w.replace('\n', '') for w in f.readlines()]

    def get_vocab(self):
        if len(self.vocab) == 0:
            self.set_vocab()
        return self.vocab

    def get_vocab_dicts(self):
        """
        Return word2id, id2word of the vocab
        """
        if len(self.vocab) == 0:
            self.set_vocab()

        word2id = {}
        id2word = {}
        i = 0
        for word in self.vocab:
            word2id[word] = i
            id2word[i] = word
            i += 1
        return word2id, id2word
