from net.multimodal.data_provider.experiment_data import EvaluationData

from net.multimodal.data_provider.data_tests import test_data_config
from net.multimodal import multimodal_utils

json_fname = test_data_config.exp_config['json_path_test']
cnn_fname = test_data_config.exp_config['cnn_regions_path_test']
imgid2region_indices = multimodal_utils.mk_toy_img_id2region_indices(json_fname,
                                                                     num_regions_per_img=5,
                                                                     subset_num_items=-1)
w2v_vocab_fname = test_data_config.exp_config['word2vec_vocab']
w2v_vectors_fname = test_data_config.exp_config['word2vec_vectors']
external_vocab_fname = test_data_config.exp_config['external_vocab']

eval_data = EvaluationData(json_fname, cnn_fname, imgid2region_indices,
                           w2v_vocab_fname, w2v_vectors_fname,
                           external_vocab_fname, subset_num_items=10)


print eval_data.X_img.shape
print eval_data.X_txt.shape
print eval_data.y.shape
print eval_data.y_true_img2txt.shape

# TODO: extend tests with asserts