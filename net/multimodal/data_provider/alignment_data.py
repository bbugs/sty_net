import numpy as np
from cs231n.multimodal.data_provider.json_data import JsonFile
from cs231n.multimodal.data_provider.vocab_data import Vocabulary


class AlignmentData(object):

    def __init__(self, json_file, cnn_file, num_regions_per_img):
        self.img_id2cnn_region_index = {}  # dict[img_id]= list of region indices of cnn file

        # self.cnn_region_index_from_img_id


class AlignmentDataV0(object):

    def __init__(self, data_config, split, num_items):
        self.d = data_config
        self.split = split
        if split == 'train':
            fname = self.d['json_path_train']
        elif split == 'val':
            fname = self.d['json_path_val']
        elif split == 'test':
            fname = self.d['json_path_test']
        else:
            raise ValueError("only train, val and test splits supported")
        self.json_file = JsonFile(fname, num_items)
        self.img_ids_in_split = self.json_file.get_img_ids()

        return

    @staticmethod
    def make_region2pair_id(img_ids, num_regions_per_img):
        region2pair_id = np.zeros((len(img_ids) * num_regions_per_img, ), dtype=int)

        region_index = 0
        n = len(img_ids)
        for i in range(n):
            region2pair_id[region_index: region_index + num_regions_per_img] = i
            region_index += num_regions_per_img

        return region2pair_id

    def make_word2pair_id(self, img_ids, verbose=False):
        word2pair_id = np.array([])  # empty array

        counter = 0
        for img_id in img_ids:
            assert img_id in set(self.img_ids_in_split)  # check that img_id is in split (json file)
            unique_word_list = self.json_file.get_word_list_of_img_id(img_id)

            n_words = len(unique_word_list)
            if verbose:
                print "n words", n_words
                print unique_word_list
            pair_ids = counter * np.ones(n_words, dtype=int)

            word2pair_id = np.hstack((word2pair_id, pair_ids))
            counter += 1

        return word2pair_id

    @staticmethod
    def pair_id2y(region2pair_id, word2pair_id):
        """
        This y only encodes whether the region and the
        word occur together in the dataset.  This is a "block y".
        """

        N = np.max(region2pair_id)
        assert N == np.max(word2pair_id)

        n_regions = region2pair_id.shape[0]
        n_words = word2pair_id.shape[0]
        y = -np.ones((n_regions, n_words), dtype=int)

        for i in range(N + 1):
            MEQ = np.outer(region2pair_id == i, word2pair_id == i)
            y[MEQ] = 1

        return y

    def make_y_true_txt2img(self, num_regions_per_img):
        """
        zappos vocabulary are the queries.

        y is y_true_zappos_img for txt2img

        Queries: txt from external vocab
        Target: image regions from json_file instantiated in the constructor of this class

        assume a fixed num_regions_per_img

        Return:
            - y_true_txt2img: np array of size (|external_vocab|, n_regions_per_img * n_img_in_split)

        """

        # set external (zappos) vocabulary
        v = Vocabulary(self.d['external_vocab'])  # zappos vocab
        external_vocab = v.get_vocab()  # list of words in vocab
        word2id, id2word = v.get_vocab_dicts()

        n_imgs_in_split = self.json_file.get_num_items()

        img_ids = self.json_file.get_img_ids()

        y_true_txt2img = -np.ones((len(external_vocab), n_imgs_in_split * num_regions_per_img), dtype=int)

        region_index = 0
        for img_id in img_ids:
            word_list = self.json_file.get_word_list_of_img_id(img_id)
            # keep words only from the zappos (external) vocab.  These are the queries.
            word_list_external_vocab = [w for w in word_list if w in external_vocab]
            for word in word_list_external_vocab:
                word_id = word2id[word]
                y_true_txt2img[word_id, region_index:region_index + num_regions_per_img] = 1
            region_index += num_regions_per_img

        return y_true_txt2img

    # def make_y_true_img2txt(self, num_regions_per_img, target_vocab_fname, external_only=False):
    #     """
    #     Queries are the images from json_file instantiated in the constructor of this class
    #     targets are the words from target_vocab_fname
    #
    #     The true alignment can be with the actual words
    #
    #     y is y_true_all_vocab for img2txt
    #
    #     assume a fixed num_regions_per_img
    #
    #     Return:
    #         - y_true_img2txt: np array of size (n_regions_per_img * n_img_in_split, |target_vocab|)
    #
    #     """
    #     v = Vocabulary(target_vocab_fname)
    #     target_vocab = v.get_vocab()
    #     word2id, id2word = v.get_vocab_dicts()
    #
    #     # load external vacab if needed
    #     ext_vocab = set()
    #     if external_only:
    #         ev = Vocabulary(self.d['external_vocab'])
    #         ext_vocab = set(ev.get_vocab())
    #
    #     # get the number of query images (these are images in the json file)
    #     n_imgs_in_split = self.json_file.get_num_items()
    #     n_regions = n_imgs_in_split * num_regions_per_img
    #
    #     y_true_img2txt = -np.ones((n_regions, len(target_vocab)), dtype=int)
    #
    #     region_index = 0
    #     for item in self.json_file.dataset_items:
    #         # get the text of the item
    #         word_list = self.json_file.get_word_list_from_item(item)
    #         for word in word_list:
    #             word_id = word2id[word]
    #
    #             if external_only:  # only words from external vocab are correct
    #                 if word not in ext_vocab:
    #                     continue
    #             y_true_img2txt[region_index:region_index + num_regions_per_img, word_id] = 1
    #
    #     return

    def get_img_id2cnn_region_index(self, num_regions):
        """
        To index the cnn data

        return a dict[img_id] = list of region indices
        """

        imgid2region_index = {}

        img_ids = self.img_ids_in_split

        region_index = 0
        for img_id in img_ids:
            imgid2region_index[img_id] = []
            for i in range(num_regions):
                imgid2region_index[img_id].append(region_index)
                region_index += 1
        return imgid2region_index



    # def make_y_true_img2txt_v0(self, num_regions_per_img, min_word_freq, target_split='test'):
    #     """
    #     Queries are the images from self.split
    #     targets are the words from
    #
    #     y is y_true_all_vocab for img2txt
    #
    #     assume a fixed num_regions_per_img
    #
    #     Return:
    #         - y_true_img2txt: np array of size (n_regions_per_img * n_img_in_split, |target_vocab|)
    #
    #     """
    #     # load target json file if different from self.split
    #     if target_split != self.split:
    #         # we need to load a new json
    #         if target_split == 'train':
    #             target_json_file = JsonFile(self.d['json_path_train'])
    #         elif target_split == 'val':
    #             target_json_file = JsonFile(self.d['json_path_val'])
    #         elif target_split == 'test':
    #             target_json_file = JsonFile(self.d['json_path_test'])
    #         else:
    #             raise ValueError("train, val and test supported")
    #     else:
    #         target_json_file = self.json_file
    #
    #     # get vocabulary word from target split
    #     target_vocab = target_json_file.get_vocab_words_from_json(min_word_freq=min_word_freq)
    #
    #     # get the number of query images (these are images in self.split)
    #     n_imgs_in_split = self.json_file.get_num_items()
    #     n_regions = n_imgs_in_split * num_regions_per_img
    #
    #     y_true_img2txt = -np.ones((n_regions, len(target_vocab)))
    #
    #     region_index = 0
    #     for item in self.json_file.dataset['items']:
    #         # get the text of the item
    #         word_list = self.json_file.get_word_list_from_item(item)
    #
    #
    #     return

