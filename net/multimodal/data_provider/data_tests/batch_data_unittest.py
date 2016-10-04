from net.multimodal.data_provider.batch_class import BatchData

from net.multimodal.data_provider.data_tests import test_data_config
from net.multimodal import multimodal_utils

json_fname = test_data_config.exp_config['json_path_test']
cnn_fname = test_data_config.exp_config['cnn_regions_path_test']
imgid2region_indices = multimodal_utils.mk_toy_img_id2region_indices(json_fname,
                                                                     num_regions_per_img=5,
                                                                     subset_num_items=-1)
w2v_vocab_fname = test_data_config.exp_config['word2vec_vocab']
w2v_vectors_fname = test_data_config.exp_config['word2vec_vectors']

batch_data = BatchData(json_fname, cnn_fname, imgid2region_indices, w2v_vocab_fname, w2v_vectors_fname)

X_img, X_txt, region2pair_id, word2pair_id = batch_data.get_minibatch(batch_size=5, verbose=True)


X_img2, X_txt2, region2pair_id2, word2pair_id2 = batch_data.get_minibatch(batch_size=5, verbose=True)

X_img3, X_txt3, region2pair_id3, word2pair_id3 = batch_data.get_minibatch(batch_size=5, verbose=True)

X_img4, X_txt4, region2pair_id4, word2pair_id4 = batch_data.get_minibatch(batch_size=5, verbose=True)
# TODO: extend tests with asserts