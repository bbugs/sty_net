from net.multimodal.data_provider.data_tests import test_data_config
from net.multimodal.data_provider.json_data import JsonFile, check_img_ids
from net.multimodal import multimodal_utils

d = test_data_config.exp_config
json_file = JsonFile(d['json_path_test'], num_items=50)
print "\nrandom img ids"
print json_file.get_random_img_ids(num_imgs=4)

print "\nrandom img ids"
print json_file.get_random_img_ids(num_imgs=4)

print "\nrandom img ids"
print json_file.get_random_img_ids(num_imgs=4)

print "\nrandom img ids"
print json_file.get_random_img_ids(num_imgs=4)

print "\nrandom img ids"
print json_file.get_random_img_ids(num_imgs=4)
