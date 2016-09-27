"""
Test word2img_ids_index

"""
from net.multimodal.data_provider import json_data
num_items = 20
json_fname = "../data/fashion53k/json/with_ngrams/dataset_dress_all_test.clean.json"
json_file = json_data.JsonFile(json_fname=json_fname, num_items=num_items)  #todo: change to -1

word2img_ids_index = json_file.get_word2img_ids_index(remove_stops=True, min_word_freq=5,
                                                      save_fname=None)

# print word2img_ids_index
print word2img_ids_index
