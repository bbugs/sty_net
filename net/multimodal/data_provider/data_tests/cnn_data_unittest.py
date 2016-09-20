
from cs231n.multimodal.data_provider.cnn_data import CnnData, check_num_regions
from cs231n.multimodal.data_provider.data_tests import test_data_config
from cs231n.multimodal import multimodal_utils
import numpy as np

from cs231n.multimodal import multimodal_utils

json_fname = test_data_config.exp_config['json_path_test']
cnn_fname = test_data_config.exp_config['cnn_regions_path_test']
cnn_data = CnnData(cnn_fname=cnn_fname)

cnn = cnn_data.get_cnn_from_index(index=5)

correct = np.array(
    [0, 0, 0, 4.8477, 0, 0, 0, 0, 0, 0, 0,  # 11
     1.96109, 0, 0, 0, 0, 0, 0, 0,  # 8
     2.70587, 0, 0, 0, 0, 0.129899,  # 6
     0, 0, 0.8699]  # 3
)  # 28

assert np.allclose(cnn[0:28], correct)

# print cnn_data.get_cnn_dim()  # 4096
assert cnn_data.get_cnn_dim() == 4096


imgid2region_indices = multimodal_utils.mk_toy_img_id2region_indices(json_fname=json_fname,
                                                                     num_regions_per_img=5,
                                                                     subset_num_items=-1)

check_num_regions(cnn_fname, imgid2region_indices, verbose=True)


