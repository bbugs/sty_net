from net.multimodal import multimodal_utils
import numpy as np

print multimodal_utils.init_random_weights(3,4)
print multimodal_utils.init_random_weights(4).shape


json_fname = '../data/fashion53k/json/only_zappos/dataset_dress_all_test.clean.json'
num_regions_per_img = 1
img_id2region_indices_1 = multimodal_utils.mk_toy_img_id2region_indices(json_fname,
                                                                      num_regions_per_img,
                                                                      subset_num_items=-1)

assert img_id2region_indices_1[6] == [0]
assert img_id2region_indices_1[80] == [1]
print img_id2region_indices_1

num_regions_per_img = 5
img_id2region_indices_5 = multimodal_utils.mk_toy_img_id2region_indices(json_fname,
                                                                      num_regions_per_img,
                                                                      subset_num_items=-1)

assert img_id2region_indices_5[6] == [0, 1, 2, 3, 4]
assert img_id2region_indices_5[80] == [5, 6, 7, 8, 9]

print img_id2region_indices_5


cnn_region_index2img_id = multimodal_utils.mk_cnn_region_index2img_id(img_id2region_indices_1)

print cnn_region_index2img_id[0]
assert cnn_region_index2img_id[0] == 6
assert cnn_region_index2img_id[1] == 80


cnn_region_index2img_id = multimodal_utils.mk_cnn_region_index2img_id(img_id2region_indices_5)

print cnn_region_index2img_id[0]
assert cnn_region_index2img_id[0] == 6
assert cnn_region_index2img_id[5] == 80

# it returns diferent arrays every time it's called
# print np.random.randn(3,4)
