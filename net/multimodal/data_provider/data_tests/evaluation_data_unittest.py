from net.multimodal.data_provider.eval_data_class import EvaluationData

from net.multimodal import multimodal_utils
import numpy as np

json_fname = '../data/fashion53k/json/only_zappos/dataset_dress_all_test.clean.json'
cnn_fname = '../data/fashion53k/full_img/per_split/cnn_fc7_test.txt'
imgid2region_indices = multimodal_utils.mk_toy_img_id2region_indices(json_fname, num_regions_per_img=1,
                                                                     subset_num_items=-1)

w2v_vocab_fname = '../data/word_vects/glove/vocab.txt'
w2v_vectors_fname = '../data/word_vects/glove/vocab_vecs.txt'
external_vocab_fname = '../data/fashion53k/external_vocab/zappos.vocab.txt'

eval_data = EvaluationData(json_fname, cnn_fname, imgid2region_indices,
                           w2v_vocab_fname, w2v_vectors_fname,
                           external_vocab_fname, subset_num_items=10)

print "X_img \n", eval_data.X_img
print "X_txt \n", eval_data.X_txt
print "y \n", eval_data.y  # (num_regions, len(external_vocab_words))
print "img_id2word_ids_ext_vocab \n", eval_data.img_id2word_ids_ext_vocab
print "ext_vocab_word2img_ids \n", eval_data.ext_vocab_word2img_ids
print "external_vocab_words \n", eval_data.external_vocab_words
print "ext_vocab_word2id \n", eval_data.ext_vocab_word2id


######################################
# Test shapes
######################################

V = len(eval_data.external_vocab_words)
print "External vocab size", V  # 214 zappos words
assert len(eval_data.img_ids) == 10  # there are 10 imgs
assert eval_data.num_regions_in_split == 10  # there are 10 imgs
assert len(eval_data.true_words_list) == 10  # there are 10 imgs
assert len(eval_data.true_words_list[0]) == 2  # first img has only 2 words
assert len(eval_data.true_word_ids_list[0]) == 2  # first img has only 2 words
assert eval_data.X_img.shape == (10, 4096)  # there are 10 imgs and 4096 cnn features
assert eval_data.X_txt.shape == (V, 200)  # V rows and 200 w2v dim
assert eval_data.y.shape == (10, V)  # num regions, V
assert len(eval_data.img_id2word_ids_ext_vocab) == 10
# assert len(eval_data.ext_vocab_word2id) == V  # does not equal V because the dict is incomplete, but this is ok.

######################################
# Test y
######################################
one_shoulder_index = eval_data.external_vocab_words.index("one-shoulder")
# one-sholder occurs only in the second image and not in the other two (first and third)
assert np.allclose(eval_data.y[1, one_shoulder_index], 1)  # second img
assert np.allclose(eval_data.y[0, one_shoulder_index], -1)  # first img
assert np.allclose(eval_data.y[2, one_shoulder_index], -1)  # third img

v_neck_index = eval_data.external_vocab_words.index("v-neck")
assert np.allclose(eval_data.y[1, v_neck_index], -1)  # second img
assert np.allclose(eval_data.y[0, v_neck_index], -1)  # first img
assert np.allclose(eval_data.y[2, v_neck_index], 1)   # third img

######################################
# Test external_vocab_words and ext_vocab_word2id
######################################
dress_index = eval_data.external_vocab_words.index("dress")
assert eval_data.ext_vocab_word2id["dress"] == dress_index
bridesmaid_index = eval_data.external_vocab_words.index("bridesmaid")
assert eval_data.ext_vocab_word2id["bridesmaid"] == bridesmaid_index

######################################
# Test img_id2word_ids
######################################
assert eval_data.img_id2words_ext_vocab[6] == ['bridesmaid', 'dress']
assert eval_data.img_id2word_ids_ext_vocab[6] == [14, 9]

print "Tests completed"