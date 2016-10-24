"""

The purpose is to preprocess both the json files and the zappos vocabulary to include only words that are in the w2v vocabulary.

Otherwise, there is no wordvector to work with and the network might just get really confused.

Preprocess zappos vocab

Load a file with the zappos vocabulary
Load the train json file that has already been cleaned with a zappos vocabulary
Load a word2vec vocab file

Compute the frequencies of the zappos vocabulary using the json file
discard the words that occur less than 5 times
discard the words that are not in the word2vec vocab file

Write a new zappos vocab file
Write a new json file with only these words

Perhaps write a zappos vocab for test purposes only????  Otherwise there are some words that simply have no relevant images because they do not appear in the test set.

"""

import json
from collections import defaultdict
import os

from net.multimodal.data_provider.vocab_data import Vocabulary
from net.multimodal.data_provider.json_data import JsonFile

# TODO: what to do with images that have now words.  IGNORE


class DataBundleCreator(object):

    def __init__(self, rpath_bundle, min_n_imgs_per_word=100,
                 zappos_vocab_fname='',
                 w2v_vocab_fname='',
                 json_rpath='',
                 verbose=False):

        self.rpath_bundle = rpath_bundle
        self.min_n_imgs_per_word = min_n_imgs_per_word
        self.zappos_vocab_fname = zappos_vocab_fname
        self.w2v_vocab_fname = w2v_vocab_fname
        self.json_rpath = json_rpath
        self.verbose = verbose

        self._mk_bundle_dir()
        self._read_zappos_vocab()
        self._load_w2v_vocab()
        self._filter_vocab_on_freq()
        self._filter_vocab_on_w2v()
        self._save_new_vocab()
        self._save_json(split='train')
        self._save_json(split='val')
        self._save_json(split='test')

    def _mk_bundle_dir(self):
        if not os.path.isdir(self.rpath_bundle):
            os.mkdir(self.rpath_bundle)

    def _read_zappos_vocab(self):
        # Read zappos vocabulary file
        zv = Vocabulary(fname=self.zappos_vocab_fname)
        self.zappos_vocab = set(zv.get_vocab())

        if self.verbose:
            print "zappos vocab len \n", len(self.zappos_vocab)

    def _load_w2v_vocab(self):
        w2v_v = Vocabulary(fname=self.w2v_vocab_fname)
        self.w2v_vocab = set(w2v_v.get_vocab())
        if self.verbose:
            print "w2v vocab len \n", len(self.w2v_vocab)

            print "number of common words between zappos and w2v:", \
                len(self.zappos_vocab.intersection(self.w2v_vocab))

    def _get_json_file(self, split):
        json_fname = self.json_rpath + '/dataset_dress_all_{}.clean.json'.format(split)
        return JsonFile(json_fname=json_fname, num_items=-1)

    def _get_word2img_ids(self):
        # print "len vocab from json \n", len(json_vocab)  # 197 min freq 5
        json_file = self._get_json_file(split='train')  # only train because we filter based on train split
        json_file.set_img_ids()
        word2img_ids = json_file.get_word2img_ids_index(remove_stops=True,
                                                             min_word_freq=0)  # get all words
        if self.verbose:
            print "len of word2img_ids \n", len(word2img_ids)  #
        return word2img_ids

    def _filter_vocab_on_freq(self):
        word2img_ids = self._get_word2img_ids()

        # compute how many images per word
        word2n_imgs = defaultdict(int)
        for word in word2img_ids:
            word2n_imgs[word] = len(word2img_ids[word])
        self.freq_filtered_vocab = []
        for w in sorted(word2n_imgs, key=word2n_imgs.get, reverse=True):
            # print w, n_imgs_per_word[w]
            if word2n_imgs[w] >= self.min_n_imgs_per_word:
                self.freq_filtered_vocab.append(w)

    def _filter_vocab_on_w2v(self):
        self.new_vocab = [w for w in self.freq_filtered_vocab if w in self.w2v_vocab]

    def _save_new_vocab(self):
        # Save the new vocabulary file
        print "saving new vocabulary"
        new_vocab_fname = self.rpath_bundle + '/zappos.w2v.vocab.txt'
        with open(new_vocab_fname, 'w') as f:
            for w in self.new_vocab:
                f.write('{}\n'.format(w))
        if self.verbose:
            print "number of zappos vocabulary after filtering", len(self.new_vocab)

    def _save_json(self, split):
        if split == 'train':
            json_fname = self.json_rpath + '/dataset_dress_all_{}.clean.json'.format(split)
        elif split == 'val':
            json_fname = self.json_rpath + '/dataset_dress_all_{}.clean.json'.format(split)
        elif split == 'test':
            json_fname = self.json_rpath + '/dataset_dress_all_{}.clean.json'.format(split)
        else:
            raise ValueError("")

        new_json_fname = self.rpath_bundle + '{}.json'.format(split)
        json_file = JsonFile(json_fname=json_fname, num_items=-1)

        if self.verbose:
            print "saving json {}".format(split)
        new_data = {}
        new_data['items'] = []

        for item in json_file.dataset_items:
            new_item = item
            img_id = item['imgid']
            words = json_file.get_word_list_of_img_id(img_id, remove_stops=True)
            new_words = [w for w in words if w in self.new_vocab]
            new_item['text'] = ' '.join(new_words)
            new_data['items'].append(item)

        with open(new_json_fname, 'w') as outfile:
            json.dump(new_data, outfile, indent=4)


# min_n_imgs_per_word = 100
#
# # save data here
# rpath = '../data/fashion53k/data_bundle_{}/'.format(min_n_imgs_per_word)
# if not os.path.isdir(rpath):
#     os.mkdir(rpath)
# new_vocab_fname = rpath + '/zappos.w2v.true.txt'.format(min_n_imgs_per_word)
# new_json_train_fname = rpath + 'train.json'
# new_json_val_fname = rpath + 'val.json'
# new_json_test_fname = rpath + 'test.json'

# # Read zappos vocabulary file
# zv = Vocabulary(fname='../data/fashion53k/external_vocab/zappos.vocab.txt')
# zappos_vocab = set(zv.get_vocab())
# print "zappos vocab len \n", len(zappos_vocab)

# Load train json file
# json_fname = '../data/fashion53k/json/only_zappos/dataset_dress_all_train.clean.json'
# json_train = JsonFile(json_fname=json_fname, num_items=-1)
# json_train.set_img_ids()

# print "len vocab from json \n", len(json_vocab)  # 197 min freq 5
# word2img_ids = json_train.get_word2img_ids_index(remove_stops=True,
#                                                  min_word_freq=0)  # initially get all words
# print "len of word2img_ids \n", len(word2img_ids)  #
#
# # see number of images per words
# n_imgs_per_word = defaultdict(int)
# for word in word2img_ids:
#     n_imgs_per_word[word] = len(word2img_ids[word])
#
#
# # Filter vocabulary based on min number of images per word
# filtered_vocab = []
# for w in sorted(n_imgs_per_word, key=n_imgs_per_word.get, reverse=True):
#     # print w, n_imgs_per_word[w]
#     if n_imgs_per_word[w] >= min_n_imgs_per_word:
#         filtered_vocab.append(w)
#
# # Now filter based on word2vec words available
# w2v_v = Vocabulary(fname='../data/word_vects/glove/vocab.txt')
# w2v_vocab = set(w2v_v.get_vocab())
# print "w2v vocab len \n", len(w2v_vocab)
#
# new_filter_vocab = [w for w in filtered_vocab if w in w2v_vocab]
#
#
# # Save the new vocabulary file
# print "saving new vocabulary"
# with open(new_vocab_fname, 'w') as f:
#     for w in new_filter_vocab:
#         f.write('{}\n'.format(w))
#
#
# # clean up train json file with new_filter_vocab
# print "saving json"
# new_data = {}
# new_data['items'] = []
# for item in json_train.dataset_items:
#     new_item = item
#     img_id = item['imgid']
#     words = json_train.get_word_list_of_img_id(img_id, remove_stops=True)
#     new_words = [w for w in words if w in new_filter_vocab]
#     new_item['text'] = ' '.join(new_words)
#     new_data['items'].append(item)
#
# with open(new_json_train_fname, 'w') as outfile:
#     json.dump(new_data, outfile, indent=4)
#
#
# json_fname_val = '../data/fashion53k/json/only_zappos/dataset_dress_all_val.clean.json'
# json_val = JsonFile(json_fname=json_fname_val, num_items=-1)
# json_val.set_img_ids()
#
# new_data = {}
# new_data['items'] = []
# for item in json_val.dataset_items:
#     new_item = item
#     img_id = item['imgid']
#     words = json_val.get_word_list_of_img_id(img_id, remove_stops=True)
#     new_words = [w for w in words if w in new_filter_vocab]
#     # print len(words), len(new_words)
#     new_item['text'] = item
#     new_data['items'].append(item)
#
# with open(new_json_val_fname, 'w') as outfile:
#     json.dump(new_data, outfile)
#
#
#
#
# # Read word2vec vocaublary file
#
#
#
#
#
#
#
#
# txt = json_train.get_all_txt_from_json()
# print len(json_train.get_vocabulary_words_with_counts(txt=txt, min_word_freq=5))
#
#
#
