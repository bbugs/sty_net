
from net.multimodal.data_provider.data_tests import test_data_config
from net.multimodal.data_provider.json_data import JsonFile
from net.multimodal.multimodal_utils import check_img_ids
from net.multimodal import multimodal_utils

d = test_data_config.exp_config

json_file = JsonFile(d['json_path_test'], num_items=10)

print "\nnum of items"
print json_file.get_num_items()
# 10

print "\nids from split"
print json_file.get_img_ids()
# [6, 80, 147, 212, 261, 373, 385, 431, 460, 476]

# cnn_region_ind = json_file.get_cnn_region_indeces_from_img_id(target_img_id=80, num_regions_per_img=4)
# assert cnn_region_ind == [4, 5, 6, 7]

# cnn_region_ind = json_file.get_cnn_region_indeces_from_img_id(target_img_id=1, num_regions_per_img=4)  # error returned since target_img_id not in json file

# num_regions_per_img = {}
# num_regions_per_img[80] = 2
# num_regions_per_img[373] = 6
#
# cnn_region_ind = json_file.get_cnn_region_indeces_from_img_id(target_img_id=80,
#                                                               num_regions_per_img=num_regions_per_img)

# print cnn_region_ind
# raw_input("key to cont.")


print "\nindex of img id"
print json_file.get_json_index_from_img_id(target_img_id=476)  # 9

print "\nrandom img ids"
print json_file.get_random_img_ids(num_imgs=4)
# [  6 385 373  80]  list of random img ids in the json file

# print json_file.get_index_from_img_id(1)  # returns error cause img_id 1 is not in the test split

print "\nitem of img id"
print json_file.get_item_from_img_id(target_img_id=476)
# {u'asin': u'B00EC7KR14', u'url': u'http://ecx.images-amazon.com/images/I/41FQgL4OxAL.jpg', u'text': u'vogue ...

print "\nwords of img id"
words = json_file.get_word_list_of_img_id(img_id=476, remove_stops=True)
print "\n", words
print "\nnum words of img id"
print len(words)

vocab_words = json_file.get_vocab_words_from_json(remove_stops=True, min_word_freq=0)
print "\nvocab_words"
print vocab_words
print "\nnum vocab words"
print json_file.get_num_vocab_words_from_json(remove_stops=True, min_word_freq=5)

json_file = JsonFile(d['json_path_train'], num_items=-1)
print "\nnum vocab words for the split"
print json_file.get_num_vocab_words_from_json(remove_stops=True, min_word_freq=20)

cnn_fname = d['cnn_regions_path_test']
imgid2region_indices = multimodal_utils.mk_toy_img_id2region_indices(json_fname=d['json_path_test'],
                                                                     cnn_fname=cnn_fname,
                                                                     num_regions_per_img=5,
                                                                     subset_num_items=-1)

check_img_ids(json_fname=d['json_path_test'], imgid2region_indices=imgid2region_indices)
