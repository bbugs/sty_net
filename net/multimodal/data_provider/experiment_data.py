from net.multimodal.data_provider.json_data import JsonFile, check_img_ids
from net.multimodal.data_provider.cnn_data import CnnData, check_num_regions
from net.multimodal.data_provider.vocab_data import Vocabulary
from net.multimodal.data_provider.word2vec_data import Word2VecData
import numpy as np
from net.multimodal import multimodal_utils


class ExperimentData(object):
    def __init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                 w2v_vocab_fname, w2v_vectors_fname, subset_num_items):
        """
        Inputs:
        json_fname: str
        cnn_fname: str
        batch_size: int
        img_id2cnn_region_indeces:  a dict[img_id] = list of indices of cnn regions

        """
        # check that cnn file and img_id2cnn_region_indeces are consistent
        check_num_regions(cnn_fname, img_id2cnn_region_indeces)
        # check that json file and img_id2cnn_region_indeces are consistent
        check_img_ids(json_fname, img_id2cnn_region_indeces)

        self.json_file = JsonFile(json_fname, num_items=subset_num_items)
        self.cnn_data = CnnData(cnn_fname)
        self.img_id2cnn_region_indeces = img_id2cnn_region_indeces

        self.w2v_data = Word2VecData(w2v_vocab_fname, w2v_vectors_fname)


class BatchData(ExperimentData):
    def __init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                 w2v_vocab_fname, w2v_vectors_fname, subset_num_items=-1):

        print "creating a new BatchData object"
        ExperimentData.__init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                                w2v_vocab_fname, w2v_vectors_fname, subset_num_items)

        self.cnn_dim = self.cnn_data.get_cnn_dim()
        self.w2v_dim = self.w2v_data.get_word2vec_dim()

        return

    def _reset(self):
        self.img_ids = []
        self.n_regions = 0
        self.n_imgs = 0

        self.X_img = np.array([])
        self.X_txt = np.array([])
        # self.X_txt_global = np.array([])

        self.y = np.array([])
        self.y_global = np.array([])

        self.img_ids2words = {}

        self.unique_words_list = []  # unique words
        self.n_unique_words = 0

        self.word_seq = []  # the entire sequence of words concatenated for the batch

    # self.region2pair_id = np.array([])
    # self.word2pair_id = np.array([])

    # def _mk_region2pair_id(self):
    #
    #     self.region2pair_id = np.zeros(self.n_regions, dtype=int)
    #
    #     region_index = 0
    #     i = 0
    #     for img_id in self.img_ids:
    #         n_regions_in_img_id = len(self.img_id2cnn_region_indeces[img_id])
    #         self.region2pair_id[region_index: region_index + n_regions_in_img_id] = i
    #         region_index += n_regions_in_img_id
    #         i += 1
    #
    # def _mk_word2pair_id(self):
    #
    #     counter = 0
    #     for img_id in self.img_ids:
    #         n_words_in_img = len(self.img_ids2words[img_id])
    #         pair_ids = counter * np.ones(n_words_in_img, dtype=int)
    #         self.word2pair_id = np.hstack((self.word2pair_id, pair_ids))
    #         counter += 1


    # def get_minibatch(self, batch_size, verbose=False):
    #
    #     img_ids = self.json_file.get_random_img_ids(batch_size)
    #
    #     if verbose:
    #         print "img_ids \n", img_ids
    #
    #     self._mk_region2pair_id()
    #     num_regions_in_bath = len(self.region2pair_id)
    #     X_img = np.zeros((num_regions_in_bath, self.cnn_data.get_cnn_dim()))
    #
    #     # Set word vectors for the batch X_txt and set self.word2pair_id
    #     words_in_batch = []
    #     word2pair_id = np.array([], dtype=int)  # empty array
    #     counter = 0
    #     batch_region_index = 0
    #     for img_id in img_ids:
    #         words_in_img = self.json_file.get_word_list_of_img_id(img_id, remove_stops=True)
    #         # add to self.word2pair_id
    #         n_words = len(words_in_img)
    #         pair_ids = counter * np.ones(n_words, dtype=int)
    #         word2pair_id = np.hstack((word2pair_id, pair_ids))
    #         counter += 1
    #
    #         # add words to words_in_batch
    #         words_in_batch.extend(words_in_img)
    #
    #         # Set cnn vectors for the batch X_img
    #         for region_index in self.img_id2cnn_region_indeces[img_id]:
    #             X_img[batch_region_index, :] = self.cnn_data.get_cnn_from_index(region_index)
    #             batch_region_index += 1
    #
    #     # Set word vectors for words_in_batch
    #     X_txt = self.w2v_data.get_word_vectors_of_word_list(words_in_batch)
    #
    #     return X_img, X_txt, region2pair_id, word2pair_id

    def _mk_y_local(self):
        """
        Args:
            self

        Creates:
            self.y_local:
                shape (n_regions, n_unique_words)
                y_local[i,j] = +1 if region i and word j occur together
                y_local[i,j] = -1 if region i and word j DO NOT occur together

        """
        self.y = -np.ones((self.n_regions, self.n_unique_words))
        region_index = 0
        for img_id in self.img_ids:
            n_regions_in_img = len(self.img_id2cnn_region_indeces[img_id])
            words_in_img = self.img_ids2words[img_id]

            for i in range(n_regions_in_img):
                for word in words_in_img:
                    word_index = self.unique_words_list.index(word)
                    self.y[region_index, word_index] = 1
                region_index += 1

    def _mk_X_img(self):
        self.X_img = np.zeros((self.n_regions, self.cnn_dim))

        i = 0
        for img_id in self.img_ids:
            region_indices = self.img_id2cnn_region_indeces[img_id]
            for region_index in region_indices:
                self.X_img[i, :] = self.cnn_data.get_cnn_from_index(region_index)
                i += 1
        return

    def _mk_X_txt_local(self):
        self.X_txt = self.w2v_data.get_word_vectors_of_word_list(self.unique_words_list)

    def _mk_X_txt_global(self):
        self.X_txt_global = self.w2v_data.get_word_vectors_of_word_list(self.word_seq)

    def mk_minibatch(self, batch_size, verbose=False, debug=False):
        """
        Args:
            self:
            batch_size: int
            verbose: bool

        Creates:
            self.y_local
            self.X_img
            self.X_txt_local
            self.X_txt_global

        Returns:

        """
        self._reset()

        self.img_ids = self.json_file.get_random_img_ids(batch_size)
        if debug:
            self.img_ids = sorted(self.img_ids)  # so that the unittests can expect the output in this order
        if verbose:
            print "img_ids on minibatch", self.img_ids

        # Get unique words in the batch and also get the word sequence
        unique_words = set()
        for img_id in self.img_ids:
            words_in_img = self.json_file.get_word_list_of_img_id(img_id, remove_stops=True)
            self.word_seq.extend(words_in_img)
            self.img_ids2words[img_id] = words_in_img
            unique_words.update(set(words_in_img))

        self.unique_words_list = sorted(list(unique_words))
        self.n_unique_words = len(self.unique_words_list)

        # get number of regions in the batch
        for img_id in self.img_ids:
            self.n_regions += len(self.img_id2cnn_region_indeces[img_id])

        # make y_local
        self._mk_y_local()

        # make X_img
        self._mk_X_img()

        # make X_txt_local
        self._mk_X_txt_local()

        # make X_txt_global
        # self._mk_X_txt_global()

        # make region2pair_id and word2pair_id
        # self._mk_region2pair_id()
        # self._mk_word2pair_id()

        return


class EvaluationData(ExperimentData):
    def __init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                 w2v_vocab_fname, w2v_vectors_fname, external_vocab_fname, subset_num_items=-1):

        ExperimentData.__init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                                w2v_vocab_fname, w2v_vectors_fname, subset_num_items)

        self.external_vocab = Vocabulary(external_vocab_fname)  # zappos
        self.external_vocab_words = self.external_vocab.get_vocab()  # list of words in vocab
        self.ext_vocab_word2id, self.id2word_ext_vocab = self.external_vocab.get_vocab_dicts()

        self.img_ids = self.json_file.get_img_ids()
        self.y = np.array([])  # used to be y_img2txt

        self.X_txt = np.array([])
        self.X_img = np.array([])

        self.img_id2words_ext_vocab = {}
        self.img_id2word_ids_ext_vocab = {}
        self.ext_vocab_word2img_ids = {}
        self.ext_word_id2cnn_indices = {}
        self.cnn_index2img_id = {}

        self.num_regions_in_split = 0
        self.true_words_list = []  # list of lists in the same order as img_ids, it contains the words of each img
        self.true_word_ids_list = []  # list of lists
        self.true_img_ids = []  # list of lists
        self.true_cnn_indices = []  # list of lists

        self._set_features()

    def _set_aux_dicts(self):
        self.cnn_index2img_id = multimodal_utils.mk_cnn_region_index2img_id(self.img_id2cnn_region_indeces)
        # initialize ext_vocab_word2img_ids
        for word in self.external_vocab_words:
            self.ext_vocab_word2img_ids[word] = []

        for img_id in self.img_ids:
            words_in_img = self.json_file.get_word_list_of_img_id(img_id, remove_stops=True)
            words_in_img = [w for w in words_in_img if w in self.external_vocab_words]
            self.img_id2words_ext_vocab[img_id] = words_in_img
            self.img_id2word_ids_ext_vocab[img_id] = [self.ext_vocab_word2id[w] for
                                                      w in words_in_img]
            for w in words_in_img:
                self.ext_vocab_word2img_ids[w].append(img_id)

        self._set_true_cnn_indices()

    def _set_num_regions_in_split(self):
        for img_id in self.img_ids:
            self.num_regions_in_split += len(self.img_id2cnn_region_indeces[img_id])

    def _set_true_words_list(self):

        for img_id in self.img_ids:
            words = self.img_id2words_ext_vocab[img_id]
            self.true_words_list.append(words)

    def _set_true_word_ids_list(self):

        for img_id in self.img_ids:
            word_ids = self.img_id2word_ids_ext_vocab[img_id]
            self.true_word_ids_list.append(word_ids)

    def _set_true_img_ids(self):
        for word in self.external_vocab_words:
            self.true_img_ids.append(self.ext_vocab_word2img_ids[word])

    def _set_ext_word_id2cnn_index(self):
        # the index in cnn file of correct
        for word in self.external_vocab_words:
            img_ids = self.ext_vocab_word2img_ids[word]
            word_id = self.ext_vocab_word2id[word]
            if word_id not in self.ext_word_id2cnn_indices:
                self.ext_word_id2cnn_indices[word_id] = []
            for img_id in img_ids:
                region_indices = self.img_id2cnn_region_indeces[img_id]
                for i in region_indices:
                    self.ext_word_id2cnn_indices[word_id].append(i)

    def _set_true_cnn_indices(self):
        self._set_ext_word_id2cnn_index()
        # for the set of word ids in external vocab,
        # make a list of the correct cnn indices where the word occurs
        for word_id in range(len(self.external_vocab_words)):
            cnn_indices = self.ext_word_id2cnn_indices[word_id]
            self.true_cnn_indices.append(cnn_indices)

    def _set_y(self):
        """
        y is used to evaluate classification precision and recall. It is
        not used to calculate ranking peformance
        Creates:
        self.y: Matrix of size num_regions, len(external_vocab_words). Each element
        y[i,j] = +1 or -1, indicates whether region i and word j occurred together

        """
        self.y = -np.ones((self.num_regions_in_split, len(self.external_vocab_words)), dtype=int)

        region_index = 0
        for img_id in self.img_ids:
            n_regions_in_img_id = len(self.img_id2cnn_region_indeces[img_id])

            word_ids_in_img = self.img_id2word_ids_ext_vocab[img_id]
            for word_id in word_ids_in_img:
                self.y[region_index:region_index + n_regions_in_img_id, word_id] = 1
            region_index += n_regions_in_img_id

    def _set_X_img(self):
        self.X_img = np.zeros((self.num_regions_in_split, self.cnn_data.get_cnn_dim()))

        split_region_index = 0
        for img_id in self.img_ids:
            for region_index in self.img_id2cnn_region_indeces[img_id]:
                self.X_img[split_region_index, :] = self.cnn_data.get_cnn_from_index(region_index)
                split_region_index += 1

    def _set_X_txt(self):
        self.X_txt = self.w2v_data.get_word_vectors_of_word_list(self.external_vocab_words)

    def _set_features(self):
        """
        zappos vocabulary are the queries.

        y for img2txt

        Queries: txt from external vocab
        Target: full images from json_file instantiated in the constructor of this class

        assume a fixed num_regions_per_img

        Creates:
            -y: matrix of size num_regions_in_split, len(external_vocab). It contains +1 or -1
            to indicate whether the region and the word occur together
            -X_img: matrix of cnn features. Size: n_regions x cnn_dim
            -X_txt: matrix of word2vec features. Size: n_words_in_external_vocab x w2v_dim
            -X_txt_mwq: word2vec features of multi-word-queries (mwq). Size: n_imgs x w2v_dim


        Return:
            - y_true_txt2img: np array of size (|external_vocab|, n_regions_per_img * n_img_in_split)


        """

        self._set_num_regions_in_split()
        # get external (zappos) vocabulary
        self._set_aux_dicts()
        self._set_true_words_list()
        self._set_true_word_ids_list()
        self._set_true_img_ids()
        self._set_y()
        self._set_X_img()
        self._set_X_txt()

        return


class EvaluationDataMWQ(EvaluationData):
    """
    Evaluation data for Multiple-Word Queries (MWQ)
    """

    def __init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                 w2v_vocab_fname, w2v_vectors_fname, external_vocab_fname,
                 mwq_aggregator,
                 subset_num_items=-1):

        self.aggregator = mwq_aggregator
        if mwq_aggregator == 'avg':
            self.aggregator = np.mean
        elif mwq_aggregator == 'max':
            self.aggregator = np.max
        else:
            raise ValueError("aggregator function for multiple word queries must be avg or max")

        EvaluationData.__init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                                w2v_vocab_fname, w2v_vectors_fname, external_vocab_fname,
                                subset_num_items)

    def _set_true_img_ids(self):
        for img_id in self.img_ids:
            self.true_img_ids.append([img_id])

    def _set_true_cnn_indices(self):
        self.true_cnn_indices = [[i] for i in range(self.num_regions_in_split)]


    def _set_X_txt(self):
        # set X_txt for multiple-word-queries.  This is to reproduce the textual queries
        # from the original images. We average (or max) the word2vec features to produce one query.
        print "MWQ setting X_txt"
        self.X_txt = np.zeros((len(self.img_ids), self.w2v_data.get_word2vec_dim()))
        i = 0
        for img_id in self.img_ids:
            words_in_img = self.img_id2words_ext_vocab[img_id]
            X_txt = self.w2v_data.get_word_vectors_of_word_list(words_in_img)  # (n_words_in_img, w2v_dim)

            self.X_txt[i, :] = self.aggregator(X_txt, axis=0)
            i += 1


def get_batch_data(exp_config, subset_num_items=-1):
    """
    Parameters
    ----------
    exp_config
    subset_num_items": when reading the json file, you can choose
    to get only the first subset_num_items. For experiments, it should be -1

    Returns
    -------
    BatchData object
    """
    json_fname_train = exp_config['json_path_train']
    cnn_fname_train = exp_config['cnn_regions_path_train']
    num_regions_per_img = exp_config['num_regions_per_img']  # TODO: replace this by the actual num_regions_per_img
    imgid2region_indices_train = multimodal_utils.mk_toy_img_id2region_indices(json_fname_train,
                                                                               num_regions_per_img=num_regions_per_img,
                                                                               subset_num_items=-1)
    w2v_vocab_fname = exp_config['word2vec_vocab']
    w2v_vectors_fname = exp_config['word2vec_vectors']

    batch_data = BatchData(json_fname_train, cnn_fname_train,
                           imgid2region_indices_train,
                           w2v_vocab_fname, w2v_vectors_fname,
                           subset_num_items=subset_num_items)
    return batch_data


def get_eval_data(exp_config, subset_train=-1, subset_val=-1, subset_test=-1):
    """

    Parameters
    ----------
    exp_config

    Returns
    -------
    eval_data_train
    eval_data_val
    """
    # ______________________________________________
    # Train Evaluation Data
    # ----------------------------------------------

    json_fname_train = exp_config['json_path_train']
    cnn_fname_train = exp_config['cnn_full_img_path_train']

    w2v_vocab_fname = exp_config['word2vec_vocab']
    w2v_vectors_fname = exp_config['word2vec_vectors']

    num_regions_per_img = 1  # During evaluation num_regions_per_img is set to 1 because we evaluate with full img cnn
    imgid2region_indices_train = multimodal_utils.mk_toy_img_id2region_indices(json_fname_train,
                                                                               num_regions_per_img=num_regions_per_img,
                                                                               subset_num_items=-1)
    external_vocab_fname = exp_config['external_vocab']

    eval_data_train = EvaluationData(json_fname_train, cnn_fname_train, imgid2region_indices_train,
                                     w2v_vocab_fname, w2v_vectors_fname,
                                     external_vocab_fname,
                                     subset_num_items=subset_train)
    # ______________________________________________
    # Val Evaluation Data
    # ----------------------------------------------
    print "setting evaluation data for val split"
    json_fname_val = exp_config['json_path_val']
    cnn_fname_val = exp_config['cnn_full_img_path_val']
    imgid2region_indices_val = multimodal_utils.mk_toy_img_id2region_indices(json_fname_val,
                                                                             num_regions_per_img=num_regions_per_img,
                                                                             subset_num_items=-1)

    eval_data_val = EvaluationData(json_fname_val, cnn_fname_val, imgid2region_indices_val,
                                   w2v_vocab_fname, w2v_vectors_fname,
                                   external_vocab_fname, subset_num_items=subset_val)

    # ______________________________________________
    # Test Evaluation Data
    # ----------------------------------------------
    print "setting evaluation data for test split"
    json_fname_test = exp_config['json_path_test']
    cnn_fname_test = exp_config['cnn_full_img_path_test']
    imgid2region_indices_test = multimodal_utils.mk_toy_img_id2region_indices(json_fname_test,
                                                                              num_regions_per_img=num_regions_per_img,
                                                                              subset_num_items=-1)

    eval_data_test = EvaluationData(json_fname_test, cnn_fname_test, imgid2region_indices_test,
                                    w2v_vocab_fname, w2v_vectors_fname,
                                    external_vocab_fname,
                                    subset_num_items=subset_test)

    # ______________________________________________
    # Test Evaluation Data MWQ
    # ----------------------------------------------
    print "setting evaluation data for test split MWQ"
    json_fname_test = exp_config['json_path_test']
    cnn_fname_test = exp_config['cnn_full_img_path_test']
    imgid2region_indices_test = multimodal_utils.mk_toy_img_id2region_indices(json_fname_test,
                                                                              num_regions_per_img=num_regions_per_img,
                                                                              subset_num_items=-1)

    eval_data_test_mwq = EvaluationDataMWQ(json_fname_test, cnn_fname_test, imgid2region_indices_test,
                                           w2v_vocab_fname, w2v_vectors_fname,
                                           external_vocab_fname, mwq_aggregator=exp_config['mwq_aggregator'],
                                           subset_num_items=subset_test)

    return eval_data_train, eval_data_val, eval_data_test, eval_data_test_mwq

