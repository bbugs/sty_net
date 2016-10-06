from net.multimodal.data_provider.json_data import JsonFile
from net.multimodal.multimodal_utils import check_img_ids, check_num_regions
from net.multimodal.data_provider.cnn_data import CnnData
from net.multimodal.data_provider.vocab_data import Vocabulary
from net.multimodal.data_provider.word2vec_data import Word2VecData
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import numpy as np
import random


class AssociatClassifiers(object):

    def __init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                 subset_num_items=-1):
        """
        Inputs:
        json_fname: str
        cnn_fname: str
        img_id2cnn_region_indeces:  a dict[img_id] = list of indices of cnn regions

        """
        # todo: check that there's only one region per img. More than one region is not supported for now.
        # check that cnn file and img_id2cnn_region_indeces are consistent
        check_num_regions(cnn_fname, img_id2cnn_region_indeces)
        # check that json file and img_id2cnn_region_indeces are consistent
        check_img_ids(json_fname, img_id2cnn_region_indeces)

        self.json_file = JsonFile(json_fname, num_items=subset_num_items)
        self.cnn_file = CnnData(cnn_fname)
        self.cnn_dim = self.cnn_file.get_cnn_dim()
        self.img_id2cnn_region_indeces = img_id2cnn_region_indeces

        self.split_img_ids = set(self.json_file.get_img_ids())

        self.word2img_ids_index = self.json_file.get_word2img_ids_index(remove_stops=True,
                                                                        min_word_freq=0, # create a classifier for each word, even if they are not frequent.
                                                                        save_fname=None)
        self.classifiers = {}

        return

    def _get_pos_neg_img_ids(self, word, subsample, verbose=False):
        """
        Get positive and negative images associated with given word.

        Args:
            word: str
            subsample: Bool. Indicate whether there should be
             the same number of pos neg imgs by subsampling the one with less images.
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
        negative_img_ids = self.split_img_ids - positive_img_ids

        num_positive = len(positive_img_ids)
        num_negative = len(negative_img_ids)

        if verbose:
            print "num_positive", num_positive, "num_negative", num_negative

        if subsample:  # if you want to make the len of pos and neg imgs the same:
            if num_negative < num_positive:
                if verbose:
                    print "down sample positive_img_ids for word {}\n".format(word)
                positive_img_ids = random.sample(positive_img_ids, num_negative)
            elif num_positive < num_negative:
                if verbose:
                    print "down sample negative_img_ids for word {}\n".format(word)
                negative_img_ids = random.sample(negative_img_ids, num_positive)

        return positive_img_ids, negative_img_ids

    def get_X_y(self, word, subsample, verbose):
        """
        Get X (cnn features) and y ()
        Args:
            word: string

        Returns:
            X: cnn features of positive images
            and cnn features of negative images.
            Size: (the number of imgs (possibly downsampled) associated with word, cnn_dim)
            y: +1 or -1 to inidicate positive or negative image

        """
        positive_img_ids, negative_img_ids = self._get_pos_neg_img_ids(word, subsample, verbose)

        n_rows = len(positive_img_ids) + len(negative_img_ids)
        X = np.zeros((n_rows, self.cnn_dim))
        y = np.ones(n_rows, dtype=int)

        i = 0
        for img_id in positive_img_ids:
            # get region index
            region_index = self.img_id2cnn_region_indeces[img_id]
            X[i, :] = self.cnn_file.get_cnn_from_index(region_index[0])
            i += 1
        for img_id in negative_img_ids:
            # get region index
            region_index = self.img_id2cnn_region_indeces[img_id]
            X[i, :] = self.cnn_file.get_cnn_from_index(region_index[0])
            y[i] = -1
            i += 1
        return X, y

    def fit(self, option, binarize, subsample, verbose):
        """
        Train a naive bayes classifier for each img-word pair in the dataset (json_file, cnn_file)
        Args:
            option: str: 'bernoulli' or 'multinomial'. Indicates
             what type of naive bayes classifier we want

            binarize: float. if you use bernoulli, you can binarize
            the features at a particular value. For example,
            cnn over a certain value is 1 and below is zero.

            subsample: bool.

        Creates:
            classifiers: dictionary with one classifier for each word
            classifiers[word] = clf

        """

        if option == 'bernoulli':
            assert binarize is not None

        for word in self.word2img_ids_index.keys():
            print word

            X_train, y_train = self.get_X_y(word, subsample, verbose)

            if option == 'bernoulli':
                clf = BernoulliNB(alpha=1.0, binarize=binarize, class_prior=None, fit_prior=True)
            elif option == 'multinomial':
                clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
            else:
                raise ValueError("option {} not implemented".format(option))

            clf.fit(X_train, y_train)
            self.classifiers[word] = clf
        return

    def predict_for_img_id(self, img_id, word):
        """
        Predict the probability of word given the img_id
        Args:
            img_id: int
            word: str

        Returns:
            y: +1 or -1. Size num_img_regions. Prediction on whether
            this word is likely with this image

        """
        assert img_id in self.img_id2cnn_region_indeces
        assert word in self.classifiers

        clf = self.classifiers[word]
        region_index = self.img_id2cnn_region_indeces[img_id]

        cnn = self.cnn_file.get_cnn_from_index(region_index[0]).reshape(1, self.cnn_dim)
        y = clf.predict(cnn)

        return y

    def predict_for_cnn(self, cnn, unique_words_in_batch):
        """

        Args:
            cnn: cnn feature for ONE region. size (4096,)
            unique_words_in_batch: list of unique words in the batch

        Returns:

        """
        n_words = len(unique_words_in_batch)
        y = np.ones(n_words)

        i = 0
        for word in unique_words_in_batch:
            clf = self.classifiers[word]
            y[i] = clf.predict(cnn.reshape(1, -1))
            i += 1
        return y

    def predict_for_batch(self, X_img, unique_words_in_batch):
        """

        Args:
            X_img:
            unique_words_in_batch:

        Returns:

        """
        n_regions = X_img.shape[0]  # number of rows
        n_words = len(unique_words_in_batch)
        y = np.zeros((n_regions, n_words))
        i = 0
        for word in unique_words_in_batch:
            y[:, i] = self.classifiers[word].predict(X_img)
            i += 1

        return y



