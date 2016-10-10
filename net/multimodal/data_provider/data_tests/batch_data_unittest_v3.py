from net.multimodal.data_provider.batch_class import BatchData

from net.multimodal.data_provider.data_tests import test_data_config
from net.multimodal import multimodal_utils

json_fname = test_data_config.exp_config['json_path_test']
cnn_fname = test_data_config.exp_config['cnn_regions_path_test']
imgid2region_indices = multimodal_utils.mk_toy_img_id2region_indices(json_fname,
                                                                     cnn_fname=cnn_fname,
                                                                     num_regions_per_img=5,
                                                                     subset_num_items=-1)
w2v_vocab_fname = test_data_config.exp_config['word2vec_vocab']
w2v_vectors_fname = test_data_config.exp_config['word2vec_vectors']

batch_data = BatchData(json_fname, cnn_fname, imgid2region_indices,
                       w2v_vocab_fname, w2v_vectors_fname,
                       batch_size=3,
                       subset_num_items=3)  # TODO: change to -1 in real experiments


batch_data.mk_minibatch(verbose=True, debug=True)

print batch_data.X_img
print batch_data.y

print batch_data.X_txt
print len(batch_data.unique_words_list)

# print batch_data.X_txt_global
# print len(batch_data.word_seq)

# print batch_data.region2pair_id
# print batch_data.word2pair_id


print batch_data.X_img.shape  # (15, 4096)
print batch_data.y.shape  # (15, 88)

print batch_data.X_txt.shape  # (88, 200)
print len(batch_data.unique_words_list) # 88

# print batch_data.X_txt_global.shape  # (101, 200)
# print len(batch_data.word_seq)  # 101

# print batch_data.region2pair_id.shape  # (15,)
# print batch_data.word2pair_id.shape  # (101,)
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
