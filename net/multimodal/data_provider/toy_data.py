"""
Generate toy data
"""

import numpy as np


def gen_word_seq(N, V, max_word_per_img, seed=None):
    """ (int, int, int) -> list of np arrays

    Inputs:
    - N: number of images
    - V: size of vocabulary
    - max_word_per_img

    Returns:
    - list of numpy arrays. Each element in the list
    contains an array of int of different length
    (i.e., different number of words)
    """
    if seed:
        np.random.seed(seed)

    word_seqs = []
    for i in range(N):
        # number of words in the ith image
        n_words = np.random.randint(0, max_word_per_img)
        # word ids
        words = np.random.randint(0, V, (n_words,))
        # append to list
        word_seqs.append(words)

    return word_seqs


def gen_region_cnn(N, n_region_per_img, cnn_dim, seed=None):
    """
    Returns:
        - array of size (N * n_region_per_img, cnn_dim) with cnn codes
    """

    if seed:
        np.random.seed(seed)

    regions_cnn = np.random.uniform(0, 10, (N * n_region_per_img, cnn_dim))
    regions_cnn[regions_cnn < 5] = 0
    return regions_cnn


def gen_word2vec_vectors(V, word2vec_dim, seed=None):

    if seed:
        np.random.seed(seed)

    word2vec = np.random.randn(V, word2vec_dim)

    return word2vec


def gen_X_txt(word_seq, word_embeddings, seed=None):
    """
    Inputs:
    word_seq: list of np arrays containing word ids
    word_embeddings: np array of size (V, word2vec_dim)

    """
    if seed:
        np.random.seed(seed)

    N = len(word_seq)
    V, word2vec_dim = word_embeddings.shape
    n_words = sum([w.shape[0] for w in word_seq])

    X_txt = np.zeros((n_words, word2vec_dim))
    k = 0
    for i in range(N):
        word_ids = word_seq[i]

        for word_id in word_ids:
            X_txt[k] = word_embeddings[word_id, :]
            k += 1

    return X_txt


def gen_word_to_pair_id(word_seq):
    """(list of np arrays) -> np array

    """

    N = len(word_seq)

    word2pair_id = np.array([])  # empty array

    for i in range(N):
        word_ids = word_seq[i]
        pair_ids = i * np.ones(word_ids.shape, dtype=int)

        word2pair_id = np.hstack((word2pair_id, pair_ids))

    return word2pair_id


def gen_region_to_pair_id(N, n_regions_per_img):

    region2pair_id = np.array([])

    for i in range(N):
        pair_ids = i * np.ones(n_regions_per_img, dtype=int)
        region2pair_id = np.hstack((region2pair_id, pair_ids))

    return region2pair_id

def get_y_true_all_vocab():

    return


def get_toy_data(seed=None, verbose=False, **kwargs):

    if seed:
        np.random.seed(seed)

    N = kwargs.pop('N', 10)
    V = kwargs.pop('V', 100)
    max_word_per_img = kwargs.pop('max_word_per_img', 7)

    n_region_per_img = kwargs.pop('n_region_per_img', 3)
    cnn_dim = kwargs.pop('cnn_dim', 12)
    word_embeddings_dim = kwargs.pop('word_embeddings_dim', 4)

    word_embeddings = gen_word2vec_vectors(V, word_embeddings_dim, seed)
    word_seq = gen_word_seq(N, V, max_word_per_img, seed)
    X_img = gen_region_cnn(N, n_region_per_img, cnn_dim, seed)
    X_txt = gen_X_txt(word_seq, word_embeddings, seed)

    word2pair_id = gen_word_to_pair_id(word_seq)

    region2pair_id = gen_region_to_pair_id(N, n_region_per_img)

    if verbose:
        print word_embeddings, word_seq, X_img, X_txt, word2pair_id, region2pair_id

    return word_embeddings, word_seq, X_img, X_txt, word2pair_id, region2pair_id


def main():

    N = 10
    V = 100
    max_word_per_img = 7
    seed = 102

    word_seq = gen_word_seq(N, V, max_word_per_img, seed)

    n_region_per_img = 3
    cnn_dim = 12
    word_embeddings_dim = 4

    region_cnn = gen_region_cnn(N, n_region_per_img, cnn_dim, seed)
    word_embeddings = gen_word2vec_vectors(V, word_embeddings_dim, seed)

    X_txt = gen_X_txt(word_seq, word_embeddings, seed)

    word2pair_id = gen_word_to_pair_id(word_seq)

    region2pair_id = gen_region_to_pair_id(N, n_region_per_img)

    return

if __name__ == "__main__":

    # main()

    get_toy_data(seed=102, verbose=True)

