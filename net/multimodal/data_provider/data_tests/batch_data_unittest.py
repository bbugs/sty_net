
from cs231n.multimodal.data_provider.experiment_data import BatchData

from cs231n.multimodal.data_provider.data_tests import test_data_config
from cs231n.multimodal import multimodal_utils

json_fname = test_data_config.exp_config['json_path_test']
cnn_fname = test_data_config.exp_config['cnn_regions_path_test']
imgid2region_indices = multimodal_utils.mk_toy_img_id2region_indices(json_fname,
                                                                     num_regions_per_img=5,
                                                                     subset_num_items=-1)
w2v_vocab_fname = test_data_config.exp_config['word2vec_vocab']
w2v_vectors_fname = test_data_config.exp_config['word2vec_vectors']

batch_data = BatchData(json_fname, cnn_fname, imgid2region_indices, w2v_vocab_fname, w2v_vectors_fname)

X_img, X_txt, region2pair_id, word2pair_id = batch_data.get_minibatch(batch_size=5, seed=42, verbose=True)

# TODO: extend tests with asserts