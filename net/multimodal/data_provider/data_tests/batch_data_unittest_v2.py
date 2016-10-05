"""

Test batch_data.y_local
"""

from net.multimodal.data_provider.batch_class import BatchData

from net.multimodal.data_provider.data_tests import test_data_config
from net.multimodal import multimodal_utils
import numpy as np

json_fname = test_data_config.exp_config['json_path_test']
cnn_fname = test_data_config.exp_config['cnn_regions_path_test']
imgid2region_indices = multimodal_utils.mk_toy_img_id2region_indices(json_fname,
                                                                     cnn_fname=cnn_fname,
                                                                     num_regions_per_img=5,
                                                                     subset_num_items=-1)
w2v_vocab_fname = test_data_config.exp_config['word2vec_vocab']
w2v_vectors_fname = test_data_config.exp_config['word2vec_vectors']

batch_data = BatchData(json_fname, cnn_fname, imgid2region_indices, w2v_vocab_fname, w2v_vectors_fname, subset_num_items=3)  # TODO: change subset_num_items to -1 in real experiments

batch_data.mk_minibatch(batch_size=3, verbose=True, debug=True)
# img_ids on minibatch [  6 147  80]

print "y_local shape (n_regions x n_unique_words)", batch_data.y.shape
# y_local shape (n_regions x n_unique_words) (15, 88)


one_shoulder_index = batch_data.unique_words_list.index("one-shoulder")
# one-sholder occurs only in the second image and not in the other two (first and third)
assert np.allclose(batch_data.y[5:10, one_shoulder_index], 1)  # second img regions
assert np.allclose(batch_data.y[0:5, one_shoulder_index], -1)  # first img regions
assert np.allclose(batch_data.y[10:15, one_shoulder_index], -1) # third img regions


v_neck_index = batch_data.unique_words_list.index("v-neck")
assert np.allclose(batch_data.y[0:5, v_neck_index], -1)  # second img regions
assert np.allclose(batch_data.y[5:10, v_neck_index], -1)  # first img regions
assert np.allclose(batch_data.y[10:15, v_neck_index], 1)  # third img regions

print "tests completed. y_local looks good"

#
# print ba
