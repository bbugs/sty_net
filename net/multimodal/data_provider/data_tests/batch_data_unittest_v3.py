
from net.multimodal.data_provider.experiment_data import BatchData

from net.multimodal.data_provider.data_tests import test_data_config
from net.multimodal import multimodal_utils

json_fname = test_data_config.exp_config['json_path_test']
cnn_fname = test_data_config.exp_config['cnn_regions_path_test']
imgid2region_indices = multimodal_utils.mk_toy_img_id2region_indices(json_fname,
                                                                     num_regions_per_img=5,
                                                                     subset_num_items=-1)
w2v_vocab_fname = test_data_config.exp_config['word2vec_vocab']
w2v_vectors_fname = test_data_config.exp_config['word2vec_vectors']

batch_data = BatchData(json_fname, cnn_fname, imgid2region_indices,
                       w2v_vocab_fname, w2v_vectors_fname,
                       subset_num_items=3)  # TODO: change to -1 in real experiments


batch_data.mk_minibatch(batch_size=3, verbose=True, debug=True)

print batch_data.X_img
print batch_data.y_local

print batch_data.X_txt_local
print len(batch_data.unique_words_list)

print batch_data.X_txt_global
print len(batch_data.word_seq)

print batch_data.region2pair_id
print batch_data.word2pair_id


print batch_data.X_img.shape
print batch_data.y_local.shape

print batch_data.X_txt_local.shape
print len(batch_data.unique_words_list)

print batch_data.X_txt_global.shape
print len(batch_data.word_seq)

print batch_data.region2pair_id.shape
print batch_data.word2pair_id.shape
# TODO: VERY IMPORTANT!!! TEST all of these properly
# tests
# y_local
# X_txt_local
# X_txt_global
# region2pair_id
# word2pair_id
#
#
#
#
