import numpy as np

from net.multimodal import multimodal_utils
from net.multimodal.data_provider.experiment_data import ExperimentData
from net.multimodal.data_provider.vocab_data import Vocabulary


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


def get_eval_data(exp_config):
    """

    Parameters
    ----------
    exp_config

    Returns
    -------
    eval_data_train
    eval_data_val
    """
    eval_subset_train = exp_config['eval_subset_train']
    eval_subset_val= exp_config['eval_subset_val']
    eval_subset_test = exp_config['eval_subset_test']

    # ______________________________________________
    # Train Evaluation Data
    # ----------------------------------------------

    json_fname_train = exp_config['json_path_train']
    cnn_fname_train = exp_config['cnn_full_img_path_train']

    w2v_vocab_fname = exp_config['word2vec_vocab']
    w2v_vectors_fname = exp_config['word2vec_vectors']

    num_regions_per_img = 1  # During evaluation num_regions_per_img is set to 1 because we evaluate with full img cnn
    imgid2region_indices_train = multimodal_utils.mk_toy_img_id2region_indices(json_fname_train,
                                                                               cnn_fname=cnn_fname_train,
                                                                               num_regions_per_img=num_regions_per_img,
                                                                               subset_num_items=-1)
    external_vocab_fname = exp_config['external_vocab']

    eval_data_train = EvaluationData(json_fname_train, cnn_fname_train, imgid2region_indices_train,
                                     w2v_vocab_fname, w2v_vectors_fname,
                                     external_vocab_fname,
                                     subset_num_items=eval_subset_train)
    # ______________________________________________
    # Val Evaluation Data
    # ----------------------------------------------
    print "setting evaluation data for val split"
    json_fname_val = exp_config['json_path_val']
    cnn_fname_val = exp_config['cnn_full_img_path_val']
    imgid2region_indices_val = multimodal_utils.mk_toy_img_id2region_indices(json_fname_val,
                                                                             cnn_fname=cnn_fname_val,
                                                                             num_regions_per_img=num_regions_per_img,
                                                                             subset_num_items=-1)

    eval_data_val = EvaluationData(json_fname_val, cnn_fname_val, imgid2region_indices_val,
                                   w2v_vocab_fname, w2v_vectors_fname,
                                   external_vocab_fname, subset_num_items=eval_subset_val)

    # ______________________________________________
    # Test Evaluation Data
    # ----------------------------------------------
    print "setting evaluation data for test split"
    json_fname_test = exp_config['json_path_test']
    cnn_fname_test = exp_config['cnn_full_img_path_test']
    imgid2region_indices_test = multimodal_utils.mk_toy_img_id2region_indices(json_fname_test,
                                                                              cnn_fname=cnn_fname_test,
                                                                              num_regions_per_img=num_regions_per_img,
                                                                              subset_num_items=-1)

    eval_data_test = EvaluationData(json_fname_test, cnn_fname_test, imgid2region_indices_test,
                                    w2v_vocab_fname, w2v_vectors_fname,
                                    external_vocab_fname,
                                    subset_num_items=eval_subset_test)

    # ______________________________________________
    # Test Evaluation Data MWQ
    # ----------------------------------------------
    print "setting evaluation data for test split MWQ"
    json_fname_test = exp_config['json_path_test']
    cnn_fname_test = exp_config['cnn_full_img_path_test']
    imgid2region_indices_test = multimodal_utils.mk_toy_img_id2region_indices(json_fname_test,
                                                                              cnn_fname=cnn_fname_test,
                                                                              num_regions_per_img=num_regions_per_img,
                                                                              subset_num_items=-1)

    eval_data_test_mwq = EvaluationDataMWQ(json_fname_test, cnn_fname_test, imgid2region_indices_test,
                                           w2v_vocab_fname, w2v_vectors_fname,
                                           external_vocab_fname, mwq_aggregator=exp_config['mwq_aggregator'],
                                           subset_num_items=eval_subset_test)

    return eval_data_train, eval_data_val, eval_data_test, eval_data_test_mwq
