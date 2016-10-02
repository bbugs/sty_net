"""
Test img_id2words

"""
from net.multimodal.data_provider import json_data
num_items = 20
json_fname = "../data/fashion53k/json/with_ngrams/dataset_dress_all_test.clean.json"
json_file = json_data.JsonFile(json_fname=json_fname, num_items=num_items)  #todo: change to -1

img_id2words = json_file.get_img_id2words(remove_stops=True, min_word_freq=0)

print img_id2words


json_fname = "../data/fashion53k/json/only_zappos/dataset_dress_all_test.clean.json"
json_file = json_data.JsonFile(json_fname=json_fname, num_items=num_items)  #todo: change to -1

img_id2words = json_file.get_img_id2words(remove_stops=True, min_word_freq=0)

print img_id2words
# TODO: finish tests with asserts