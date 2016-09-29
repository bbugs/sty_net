from net.multimodal.data_provider.experiment_data import EvaluationData

from net.multimodal.data_provider.data_tests import test_data_config
from net.multimodal import multimodal_utils

json_fname = '../data/fashion53k/json/only_zappos/dataset_dress_all_test.clean.json'
cnn_fname = '../data/fashion53k/full_img/per_split/cnn_fc7_test.txt'
imgid2region_indices = multimodal_utils.mk_toy_img_id2region_indices(json_fname,
                                                                     num_regions_per_img=1,
                                                                     subset_num_items=-1)

w2v_vocab_fname = '../data/word_vects/glove/vocab.txt'
w2v_vectors_fname = '../data/word_vects/glove/vocab_vecs.txt'
external_vocab_fname = '../data/fashion53k/external_vocab/zappos.vocab.txt'

eval_data = EvaluationData(json_fname, cnn_fname, imgid2region_indices,
                           w2v_vocab_fname, w2v_vectors_fname,
                           external_vocab_fname, subset_num_items=10)


print eval_data.X_img
print eval_data.X_txt
print eval_data.y
print eval_data.X_txt_mwq
print eval_data.img_id2word_ids_ext_vocab
print eval_data.ext_vocab_word2img_ids

# TODO: extend tests with asserts