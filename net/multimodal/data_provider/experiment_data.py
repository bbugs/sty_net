from cs231n.multimodal.data_provider.json_data import JsonFile, check_img_ids
from cs231n.multimodal.data_provider.cnn_data import CnnData, check_num_regions
from cs231n.multimodal.data_provider.vocab_data import Vocabulary
from cs231n.multimodal.data_provider.word2vec_data import Word2VecData
import numpy as np
from cs231n.multimodal import multimodal_utils


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

        ExperimentData.__init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                                w2v_vocab_fname, w2v_vectors_fname, subset_num_items)

        # self.X_txt = np.array([])
        # self.X_img = np.array([])
        # self.region2pair_id = np.array([])
        # self.word2pair_id = np.array([])

        return

    # def make_region2pair_id(self):
    #
    #     if isinstance(self.num_regions_per_img, int):
    #         self.region2pair_id = np.zeros((len(self.img_ids) * self.num_regions_per_img,), dtype=int)
    #
    #         region_index = 0
    #         n = len(self.img_ids)
    #         for i in range(n):
    #             self.region2pair_id[region_index: region_index + self.num_regions_per_img] = i
    #             region_index += self.num_regions_per_img
    #
    #     elif isinstance(self.num_regions_per_img, dict):
    #         # self.num_regions_per_img[img_id] = list of indices in the cnn file
    #         region_index = 0
    #         i = 0
    #         for img_id in self.img_ids:
    #             n_regions_in_img_id = self.num_regions_per_img[img_id]
    #             self.region2pair_id[region_index: region_index + n_regions_in_img_id] = i
    #             region_index += n_regions_in_img_id
    #             i += 1

    def make_region2pair_id(self, img_ids):

        num_regions_in_batch = 0
        for img_id in img_ids:
            num_regions_in_batch += len(self.img_id2cnn_region_indeces[img_id])

        region2pair_id = np.zeros(num_regions_in_batch, dtype=int)

        region_index = 0
        i = 0
        for img_id in img_ids:
            n_regions_in_img_id = len(self.img_id2cnn_region_indeces[img_id])
            region2pair_id[region_index: region_index + n_regions_in_img_id] = i
            region_index += n_regions_in_img_id
            i += 1

        return region2pair_id

    def get_minibatch(self, batch_size, verbose=False, seed=None):

        # Randomly select img_ids from json file
        img_ids = self.json_file.get_random_img_ids(batch_size, seed=seed)

        if verbose:
            print "img_ids \n", img_ids

        region2pair_id = self.make_region2pair_id(img_ids)
        num_regions_in_bath = len(region2pair_id)
        X_img = np.zeros((num_regions_in_bath, self.cnn_data.get_cnn_dim()))

        # Set word vectors for the batch X_txt and set self.word2pair_id
        words_in_batch = []
        word2pair_id = np.array([], dtype=int)  # empty array
        counter = 0
        batch_region_index = 0
        for img_id in img_ids:
            words_in_img = self.json_file.get_word_list_of_img_id(img_id)
            # add to self.word2pair_id
            n_words = len(words_in_img)
            pair_ids = counter * np.ones(n_words, dtype=int)
            word2pair_id = np.hstack((word2pair_id, pair_ids))
            counter += 1

            # add words to words_in_batch
            words_in_batch.extend(words_in_img)

            # Set cnn vectors for the batch X_img
            for region_index in self.img_id2cnn_region_indeces[img_id]:
                X_img[batch_region_index, :] = self.cnn_data.get_cnn_from_index(region_index)
                batch_region_index += 1

        # Set word vectors for words_in_batch
        X_txt = self.w2v_data.get_word_vectors_of_word_list(words_in_batch)

        return X_img, X_txt, region2pair_id, word2pair_id

        # Set cnn vectors for the batch


class EvaluationData(ExperimentData):
    def __init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                 w2v_vocab_fname, w2v_vectors_fname, external_vocab_fname, subset_num_items=-1):

        ExperimentData.__init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                                w2v_vocab_fname, w2v_vectors_fname, subset_num_items)

        self.external_vocab = Vocabulary(external_vocab_fname)  # zappos
        self.y_true_txt2img = np.array([])

        self.X_txt = np.array([])
        self.X_img = np.array([])

        self.set_features()
        self.y_true_img2txt = self.y_true_txt2img.T  #todo: remove: txt2img?
        # todo: create evaluation data where the all words are target, but only zappos are correct (leave this after the models have been trained)

    def set_features(self):
        """
        zappos vocabulary are the queries.

        y is y_true_zappos_img for txt2img

        Queries: txt from external vocab
        Target: image regions from json_file instantiated in the constructor of this class

        assume a fixed num_regions_per_img

        Return:
            - y_true_txt2img: np array of size (|external_vocab|, n_regions_per_img * n_img_in_split)

        """

        # get external (zappos) vocabulary
        external_vocab = self.external_vocab.get_vocab()  # list of words in vocab
        word2id, id2word = self.external_vocab.get_vocab_dicts()

        img_ids = self.json_file.get_img_ids()

        num_regions_in_split = 0
        for img_id in img_ids:
            num_regions_in_split += len(self.img_id2cnn_region_indeces[img_id])

        self.y_true_txt2img = -np.ones((len(external_vocab), num_regions_in_split), dtype=int)
        self.X_img = np.zeros((num_regions_in_split, self.cnn_data.get_cnn_dim()))

        region_index = 0
        split_region_index = 0
        for img_id in img_ids:
            n_regions_in_img_id = len(self.img_id2cnn_region_indeces[img_id])
            word_list = self.json_file.get_word_list_of_img_id(img_id)
            # keep words only from the zappos (external) vocab.  These are the queries.
            word_list_external_vocab = [w for w in word_list if w in external_vocab]
            for word in word_list_external_vocab:
                word_id = word2id[word]
                self.y_true_txt2img[word_id, region_index:region_index + n_regions_in_img_id] = 1
            region_index += n_regions_in_img_id

            # Set cnn vectors for the batch X_img
            for region_index in self.img_id2cnn_region_indeces[img_id]:
                self.X_img[split_region_index, :] = self.cnn_data.get_cnn_from_index(region_index)
                split_region_index += 1

        self.X_txt = self.w2v_data.get_word_vectors_of_word_list(external_vocab)

        return


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


def get_eval_data(exp_config, subset_train=-1, subset_val=-1):
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
    cnn_fname_train = exp_config['cnn_regions_path_train']

    w2v_vocab_fname = exp_config['word2vec_vocab']
    w2v_vectors_fname = exp_config['word2vec_vectors']

    num_regions_per_img = exp_config['num_regions_per_img']  # TODO: replace this by the actual num_regions_per_img
    imgid2region_indices_train = multimodal_utils.mk_toy_img_id2region_indices(json_fname_train,
                                                                               num_regions_per_img=num_regions_per_img,
                                                                               subset_num_items=-1)

    # TODO: the evaluation data for both train and val should be with cnn for the full region
    external_vocab_fname = exp_config['external_vocab']

    eval_data_train = EvaluationData(json_fname_train, cnn_fname_train, imgid2region_indices_train,
                                     w2v_vocab_fname, w2v_vectors_fname,
                                     external_vocab_fname,
                                     subset_num_items=subset_train)  # TODO: set to -1 on the real experiments
    # ______________________________________________
    # Val Evaluation Data
    # ----------------------------------------------
    print "setting evaluation data for val split"
    json_fname_val = exp_config['json_path_val']
    cnn_fname_val = exp_config['cnn_regions_path_val']
    imgid2region_indices_val = multimodal_utils.mk_toy_img_id2region_indices(json_fname_val,
                                                                             num_regions_per_img=num_regions_per_img,
                                                                             subset_num_items=-1)

    eval_data_val = EvaluationData(json_fname_val, cnn_fname_val, imgid2region_indices_val,
                                   w2v_vocab_fname, w2v_vectors_fname,
                                   external_vocab_fname, subset_num_items=subset_val)  # TODO: set to -1 on the real experiments

    return eval_data_train, eval_data_val

