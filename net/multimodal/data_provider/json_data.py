import json
import collections
import numpy as np
import os
import random
import pickle
from net.multimodal.data_provider import vocab_data


def check_img_ids(json_fname, imgid2region_indices):
    """(str, dict) ->

    imgid2region_indices is a dict whose

    """
    json_file = JsonFile(json_fname, num_items=-1)

    assert sorted(imgid2region_indices.keys()) == json_file.get_img_ids()

    return


class JsonFile(object):

    def __init__(self, json_fname, num_items=-1):
        with open(json_fname, 'r') as f:
            self.dataset = json.load(f)
            self.dataset_items = self.dataset['items']
            self.img_ids = []
            self.img_id2json_index = {}  # dict[img_id] = json_index
            self.img_id2cnn_region_index = {}  # dict[img_id]= list of region indices of cnn file
            self.word2img_ids_index = {}  # index word and list of image ids that contain the word
            if num_items > 0:  # get the first num_items if needed
                self.dataset_items = self.dataset['items'][0: num_items]

        return

    def get_num_items(self):
        return len(self.dataset_items)

    def set_img_ids(self):
        for item in self.dataset_items:
            imgid = item['imgid']
            self.img_ids.append(imgid)

    def get_img_ids(self):
        if len(self.img_ids) == 0:
            self.set_img_ids()
        return self.img_ids

    def _set_img_id2json_index(self):
        # img_indices[img_id] = index
        index = 0
        for item in self.dataset_items:
            imgid = item['imgid']
            self.img_id2json_index[imgid] = index
            index += 1

    def get_json_index_from_img_id(self, target_img_id):
        """
        return the position (index) in the json file of target_img_id
        """
        # make sure img_ids have been set
        if len(self.img_ids) == 0:
            self.set_img_ids()

        # Check if target_img_id is in the json file. Otherwise return error
        if target_img_id not in self.img_ids:
            raise ValueError("target_img_id not in json file", target_img_id)

        # make sure json indices have been set
        if len(self.img_id2json_index) == 0:
            self._set_img_id2json_index()

        return self.img_id2json_index[target_img_id]

    def get_random_img_ids(self, num_imgs):
        # make sure img_ids have been set
        if len(self.img_ids) == 0:
            self.set_img_ids()

        if num_imgs > len(self.img_ids):
            raise ValueError("reduce number of imgs, json file has {} items".format(len(self.img_ids)))

        # random_img_ids = np.random.choice(self.img_ids, num_imgs, replace=False)  DO NOT USE THIS. IT's NOT RANDOM
        random_img_ids = np.asarray(random.sample(self.img_ids, num_imgs))
        return random_img_ids

    def get_item_from_img_id(self, target_img_id):

        index = self.get_json_index_from_img_id(target_img_id)

        return self.dataset_items[index]

    def get_word_list_of_img_id(self, img_id):
        """
        return a list of unique words that correspond to the img_id
        """
        item = self.get_item_from_img_id(img_id)

        word_list = self.get_word_list_from_item(item)
        return word_list

    @staticmethod
    def get_word_list_from_item(item):
        """
        return a list of unique words that correspond to item
        """
        txt = item['text']
        word_list = sorted(list(set([w.replace('\n', "") for w in txt.split(" ") if len(w) > 0])))  # avoid empty string
        return word_list

    # def get_text_of_img_ids(self, img_ids):
    #     id2text = {}
    #     for img_id in img_ids:
    #         # get item from json
    #         item = self.get_item_from_img_id(img_id)
    #         if item is None:
    #             print img_id, " not found"
    #             continue
    #         txt = item['text']
    #         id2text[img_id] = txt
    #     return id2text

    # def get_vocab_from_json(self):
    #     vocab = set()
    #     for item in self.dataset['items']:
    #         imgid = item['imgid']
    #         word_list = self.get_word_list_of_img_id(imgid)
    #         vocab.update(word_list)
    #     return vocab

    @staticmethod
    def get_vocabulary_words_with_counts(txt, min_word_freq):
        """(str, int) -> list
        Extract the vocabulary from a string that occur more than min_word_freq.
        Return a list of the vocabulary and the frequencies.
        """

        data = txt.split()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        # keep words that occur more than min_word_freq
        top_count_pairs = [pair for pair in count_pairs if pair[1] > min_word_freq]
        return top_count_pairs

    def json2txt_file(self, fout_name):
        """
        Read json file corresponding and write a txt file with
        all the text from the json file one line for each line.
        Assume that the text on json is already clean.
        Here we lose all the info of which product belongs to what,
        but this is useful when you just want to see all
        the text, like when training an LSTM based on text alone
        """

        f = open(fout_name, 'w')

        i = 0
        for item in self.dataset_items:
            text = item['text']
            sentences = text.split('\n ')  # assume that sentences end with "\n "
            for l in sentences:
                if len(l) == 0:
                    continue
                if not l.strip().isspace():
                    f.write(l + '\n')
            i += 1

        return

    def get_all_txt_from_json(self):

        """
        Concatenate all text from json and return it.
        """

        self.json2txt_file("tmp.txt")  # save a temp file with all the text
        with open("tmp.txt", 'r') as f:
            txt = f.read()

        os.remove("tmp.txt")  # remove temp file

        return txt

    def get_vocab_words_from_json(self, min_word_freq=5):
        """
        Get vocab words from json sorted by freq
        """
        all_text = self.get_all_txt_from_json()
        vocab_with_counts = self.get_vocabulary_words_with_counts(all_text, min_word_freq)
        vocab_words = [w[0] for w in vocab_with_counts if len(w[0]) > 0]  # avoid empty string
        return vocab_words

    def get_num_vocab_words_from_json(self, min_word_freq=5):
        return len(self.get_vocab_words_from_json(min_word_freq=min_word_freq))

    def _set_word2img_ids_index(self, min_word_freq=5):

        # make sure img_ids have been set
        if len(self.img_ids) == 0:
            self.set_img_ids()

        vocab = set(self.get_vocab_words_from_json(min_word_freq=min_word_freq))

        for img_id in self.img_ids:
            words = [w for w in self.get_word_list_of_img_id(img_id) if w in vocab]
            for w in words:
                if w not in self.word2img_ids_index:
                    self.word2img_ids_index[w] = []
                self.word2img_ids_index[w].append(img_id)
        return

    def get_word2img_ids_index(self, min_word_freq=5, save_fname=None):

        if save_fname is not None and os.path.isfile(save_fname):
            return pickle.load(open(save_fname, 'rb'))

        if len(self.word2img_ids_index) == 0:
            self._set_word2img_ids_index(min_word_freq=min_word_freq)
        if save_fname is not None:
            pickle.dump(self.word2img_ids_index, open(save_fname, 'wb'))
        return self.word2img_ids_index

    # TODO: Implement:
    # get_num_tokens_in_json
    # get_avg_num_tokens_per_product


# class JsonData(object):
#
#     def __init__(self, data_config):
#         self.d = data_config
#
#         self.json_train = {}
#         self.json_val = {}
#         self.json_test = {}
#
#         return
#
#     def set_json_split(self, split='test'):
#         if split == 'train':
#             fname = self.d['cnn_regions_path_train']
#             self.json_train = np.loadtxt(fname, delimiter=',')
#         elif split == 'val':
#             fname = self.d['cnn_regions_path_val']
#             self.json_val = np.loadtxt(fname, delimiter=',')
#         elif split == 'test':
#             fname = self.d['cnn_regions_path_test']
#             self.json_test = np.loadtxt(fname, delimiter=',')
#         else:
#             raise ValueError("only train, val and test splits supported")
#         return
#
