import numpy as np

from net.multimodal import multimodal_utils as mutils
from net.multimodal.data_provider.experiment_data import ExperimentData
from pattern_mining import associat_classifiers


class BatchData(ExperimentData):
    def __init__(self, json_fname, cnn_fname,
                 img_id2cnn_region_indeces,
                 w2v_vocab_fname, w2v_vectors_fname,
                 batch_size, subset_num_items=-1):

        print "creating a new BatchData object"
        ExperimentData.__init__(self, json_fname, cnn_fname,
                                img_id2cnn_region_indeces,
                                w2v_vocab_fname, w2v_vectors_fname,
                                subset_num_items)

        self.cnn_dim = self.cnn_data.get_cnn_dim()
        self.w2v_dim = self.w2v_data.get_word2vec_dim()
        self.batch_size = batch_size

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
        self.y_associat = np.array([])

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

    def _set_batch_img_ids(self, verbose=False, debug=False):
        self.img_ids = self.json_file.get_random_img_ids(num_imgs=self.batch_size)
        self.n_imgs = len(self.img_ids)
        if debug:
            self.img_ids = sorted(self.img_ids)  # so that the unittests can expect the output in this order
        if verbose:
            print "img_ids on minibatch", self.img_ids

    def _set_num_regions_in_batch(self):
        # get number of regions in the batch
        for img_id in self.img_ids:
            self.n_regions += len(self.img_id2cnn_region_indeces[img_id])

    def _set_words_in_batch(self):
        # Get unique words in the batch and also get the word sequence
        unique_words = set()
        for img_id in self.img_ids:
            words_in_img = self.json_file.get_word_list_of_img_id(img_id, remove_stops=True)
            self.word_seq.extend(words_in_img)
            self.img_ids2words[img_id] = words_in_img
            unique_words.update(set(words_in_img))

        self.unique_words_list = sorted(list(unique_words))
        self.n_unique_words = len(self.unique_words_list)

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

    def _mk_y_associat(self):
        self.y_associat = None  # this class does not need a y_associate. It's here for consistency
        return

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

    def mk_minibatch(self, verbose=False, debug=False):
        """
        Args:
            self:
            verbose: bool

        Creates:
            self.y_local
            self.X_img
            self.X_txt_local
            self.X_txt_global

        Returns:

        """
        self._reset()
        self._set_batch_img_ids(verbose, debug)
        self._set_num_regions_in_batch()
        self._set_words_in_batch()

        # make X_img
        self._mk_X_img()

        # make X_txt_local
        self._mk_X_txt_local()

        # make y_local
        self._mk_y_local()
        # make y_associat
        self._mk_y_associat()

        # make X_txt_global
        # self._mk_X_txt_global()

        # make region2pair_id and word2pair_id
        # self._mk_region2pair_id()
        # self._mk_word2pair_id()

        return


class BatchDataAssociat(BatchData):

    def __init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                 w2v_vocab_fname, w2v_vectors_fname, classifiers,
                 batch_size, subset_num_items=-1):
        """

        Args:
            batch_size:
            json_fname: str
            cnn_fname:  str
            img_id2cnn_region_indeces: int
            w2v_vocab_fname:  str
            w2v_vectors_fname: str
            subset_num_items: int
        """

        BatchData.__init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                           w2v_vocab_fname, w2v_vectors_fname, batch_size, subset_num_items)

        self.classifiers = classifiers

        return

    def _mk_y_associat(self):
        self.y_associat = self.classifiers.predict_for_batch(X_img=self.X_img,
                                                             unique_words_in_batch=self.unique_words_list)
        return

        
def get_batch_data(exp_config):
    """
    Parameters
    ----------
    exp_config

    Returns
    -------
    BatchData object
    """
    json_fname_train = exp_config['json_path_train']
    cnn_fname_train = exp_config['cnn_regions_path_train']
    num_regions_per_img = exp_config['num_regions_per_img']  # TODO: replace this by the actual num_regions_per_img
    imgid2region_indices_train = mutils.mk_toy_img_id2region_indices(json_fname=json_fname_train,
                                                                     cnn_fname=cnn_fname_train,
                                                                     num_regions_per_img=num_regions_per_img,
                                                                     subset_num_items=-1)
    w2v_vocab_fname = exp_config['word2vec_vocab']
    w2v_vectors_fname = exp_config['word2vec_vectors']
    subset_num_items = exp_config['subset_batch_data']
    batch_size = exp_config['batch_size']

    if exp_config['word_driven_batch'] is True:
        max_n_imgs_per_word = exp_config['max_n_imgs_per_word']
        from net.multimodal.data_provider.batch_class_word_driven import BatchDataWordDriven, \
            BatchDataWordDrivenAssociat

        if exp_config['use_associat'] > 0:
            # get trained classifiers
            classifiers = associat_classifiers.get_associat_classifiers(exp_config)

            batch_data = BatchDataWordDrivenAssociat(json_fname=json_fname_train, cnn_fname=cnn_fname_train,
                                                     img_id2cnn_region_indeces=imgid2region_indices_train,
                                                     w2v_vocab_fname=w2v_vocab_fname,
                                                     w2v_vectors_fname=w2v_vectors_fname,
                                                     batch_size=batch_size,
                                                     max_n_imgs_per_word=max_n_imgs_per_word,
                                                     classifiers=classifiers,
                                                     min_word_freq=5,
                                                     subset_num_items=subset_num_items)

        else:
            batch_data = BatchDataWordDriven(json_fname=json_fname_train, cnn_fname=cnn_fname_train,
                                             img_id2cnn_region_indeces=imgid2region_indices_train,
                                             w2v_vocab_fname=w2v_vocab_fname,
                                             w2v_vectors_fname=w2v_vectors_fname,
                                             batch_size=batch_size,
                                             max_n_imgs_per_word=max_n_imgs_per_word,
                                             min_word_freq=5,
                                             subset_num_items=subset_num_items)
        return batch_data

    elif exp_config['word_drien_batch'] is False:
        if exp_config['use_associat'] > 0:
            # get trained classifiers
            classifiers = associat_classifiers.get_associat_classifiers(exp_config)

            batch_data = BatchDataAssociat(json_fname=json_fname_train, cnn_fname=cnn_fname_train,
                                           img_id2cnn_region_indeces=imgid2region_indices_train,
                                           w2v_vocab_fname=w2v_vocab_fname,
                                           w2v_vectors_fname=w2v_vectors_fname,
                                           classifiers=classifiers,
                                           batch_size=batch_size,
                                           subset_num_items=subset_num_items)

        else:
            batch_data = BatchData(json_fname=json_fname_train, cnn_fname=cnn_fname_train,
                                   img_id2cnn_region_indeces=imgid2region_indices_train,
                                   w2v_vocab_fname=w2v_vocab_fname,
                                   w2v_vectors_fname=w2v_vectors_fname,
                                   batch_size=batch_size,
                                   subset_num_items=subset_num_items)
        return batch_data
