
import numpy as np

from net.multimodal import multimodal_utils
from net.multimodal.data_provider.experiment_data import ExperimentData
import random


class WordIter(object):
    """
    Infinitely iterate through as list.
    When you reach the end of the list, restart at the beginning
    """

    def __init__(self, word_list):
        self.i = 0
        self.word_list = word_list
        self.n = len(self.word_list)

    def __iter__(self):
        return self

    def next(self):
        # reset if you already went through the entire list
        if self.i == self.n:
            self.i = 0

        if self.i < self.n:
            i = self.i
            self.i += 1
            return self.word_list[i]


class BatchDataWordDriven(ExperimentData):
    def __init__(self, json_fname, cnn_fname,
                 img_id2cnn_region_indeces,
                 w2v_vocab_fname, w2v_vectors_fname,
                 batch_size,
                 max_n_imgs_per_word,
                 min_word_freq=0,
                 subset_num_items=-1):

        """

        Args:
            json_fname:
            cnn_fname:
            img_id2cnn_region_indeces:
            w2v_vocab_fname:
            w2v_vectors_fname:
            min_word_freq:
            subsample: Bool. Indicate whether there should be
             the same number of pos neg imgs by subsampling the one with less images.
            subset_num_items:
        """
        assert batch_size <= 1
        print "creating a new BatchData object"
        ExperimentData.__init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                                w2v_vocab_fname, w2v_vectors_fname, subset_num_items)

        self.cnn_dim = self.cnn_data.get_cnn_dim()
        self.w2v_dim = self.w2v_data.get_word2vec_dim()
        self.word2img_ids_index = self.json_file.get_word2img_ids_index(remove_stops=True,
                                                                        min_word_freq=min_word_freq,
                                                                        save_fname=None)
        self.unique_words_list = sorted(self.word2img_ids_index.keys())
        self.word_iterator = WordIter(self.unique_words_list)
        self.batch_size = batch_size
        self.max_n_imgs_per_word = max_n_imgs_per_word
        return

    def _reset(self):
        self.word = ''
        self.split_img_ids = self.img_id2cnn_region_indeces.keys()

        self.X_img = np.array([])
        self.X_txt = np.array([])
        # self.X_txt_global = np.array([])

        self.y = np.array([])
        self.y_associat = np.array([])

    def _get_pos_neg_img_ids(self, word, verbose=False):
        """
        Get positive and negative images associated with given word.

        Args:
            word: str
            verbose: Bool.

        Returns:
        positive img ids: img_ids that occur with the given word
        negative img ids: img_ids that DO NOT occur with the given word

        The number of positive and negative img_ids is the same. If not,
        the function subsamples whichever has more elements so that the
        number of elements are the same for both

        """
        # img_ids that occur with word are positive examples
        positive_img_ids = set(self.word2img_ids_index[word])

        # get the rest of the images that do not occur with the word. These are negative images
        negative_img_ids = set(self.split_img_ids) - positive_img_ids

        num_positive = len(positive_img_ids)
        num_negative = len(negative_img_ids)

        if verbose:
            print "num_positive", num_positive, "num_negative", num_negative

        if num_positive > self.max_n_imgs_per_word:
            if verbose:
                print "down sample positive_img_ids for word {}\n".format(word)
            positive_img_ids = set(random.sample(positive_img_ids,
                                                 self.max_n_imgs_per_word))
        if num_negative > self.max_n_imgs_per_word:
            if verbose:
                print "down sample negative_img_ids for word {}\n".format(word)
            negative_img_ids = set(random.sample(negative_img_ids,
                                                 self.max_n_imgs_per_word))

        return positive_img_ids, negative_img_ids

    def _get_num_regions(self, img_ids):
        n_regions = 0
        for img_id in img_ids:
            n_regions += len(self.img_id2cnn_region_indeces[img_id])
        return n_regions

    def _set_X_img_y_local(self, word):

        pos_img_ids, neg_img_ids = self._get_pos_neg_img_ids(word=word)

        n_rows = 0
        n_rows += self._get_num_regions(pos_img_ids)  # number of regions of pos imgs
        n_rows += self._get_num_regions(neg_img_ids)  # number of regions of neg imgs

        self.X_img = np.zeros((n_rows, self.cnn_dim))
        self.y = -np.ones(n_rows, dtype=int).reshape(-1, 1)

        i = 0
        for img_id in pos_img_ids:
            # get region index
            region_indices = self.img_id2cnn_region_indeces[img_id]
            for region_index in region_indices:
                self.X_img[i, :] = self.cnn_data.get_cnn_from_index(region_index)
                self.y[i] = 1
                i += 1
        for img_id in neg_img_ids:
            # get region index
            region_indices = self.img_id2cnn_region_indeces[img_id]
            for region_index in region_indices:
                self.X_img[i, :] = self.cnn_data.get_cnn_from_index(region_index)
                i += 1

        return

    def _mk_X_txt_local(self, word):
        self.X_txt = self.w2v_data.get_word_vectors_of_word_list([word])

    def _mk_y_associat(self):
        self.y_associat = None
        return

    def mk_minibatch(self, verbose=False):
        """

        Args:
            batch_size:  In this context the batch size is the number
             of images per word

        Returns:

        """

        self._reset()
        self.word = self.word_iterator.next()
        self._set_X_img_y_local(self.word)
        self._mk_X_txt_local(self.word)
        self._mk_y_associat()


class BatchDataWordDrivenAssociat(BatchDataWordDriven):
    def __init__(self, json_fname, cnn_fname,
                 img_id2cnn_region_indeces,
                 w2v_vocab_fname, w2v_vectors_fname,
                 batch_size,
                 max_n_imgs_per_word,
                 classifiers,
                 min_word_freq=0,
                 subset_num_items=-1):

        BatchDataWordDriven.__init__(self, json_fname=json_fname,
                                     cnn_fname=cnn_fname,
                                     img_id2cnn_region_indeces= img_id2cnn_region_indeces,
                                     w2v_vocab_fname=w2v_vocab_fname,
                                     w2v_vectors_fname=w2v_vectors_fname,
                                     batch_size=batch_size,
                                     max_n_imgs_per_word=max_n_imgs_per_word,
                                     min_word_freq=min_word_freq,
                                     subset_num_items=subset_num_items)

        self.classifiers = classifiers

        return

    def _mk_y_associat(self):
        self.y_associat = self.classifiers.predict_for_batch(X_img=self.X_img,
                                                             unique_words_in_batch=[self.word])

        return

