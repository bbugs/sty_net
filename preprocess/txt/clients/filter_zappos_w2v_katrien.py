"""
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

from preprocess.txt.prepro_zappos_vocab import DataBundleCreator

min_n_imgs_per_word = 100
rpath_bundle = '../data/fashion53k/data_bundle_{}_katrien/'.format(min_n_imgs_per_word)

zappos_vocab_fname = '../data/fashion53k/external_vocab/zappos.vocab.txt'
dbc = DataBundleCreator(rpath_bundle=rpath_bundle,
                        min_n_imgs_per_word=min_n_imgs_per_word,
                        zappos_vocab_fname=zappos_vocab_fname,
                        w2v_vocab_fname=rpath_bundle+'w2v.vocab.txt',
                        json_rpath='../data/fashion53k/json/only_zappos/',
                        verbose=True
                        )

# zappos vocab len
# 211
# w2v vocab len
# 8410
# number of common words between zappos and w2v: 195
# len of word2img_ids
# 208
# saving new vocabulary
# number of zappos vocabulary after filtering 140
# saving json train
# saving json val
# saving json test



