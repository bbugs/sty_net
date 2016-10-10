import net.multimodal.multimodal_utils
from net.multimodal.data_provider.eval_data_class import EvaluationData
from net.multimodal.data_provider import cnn_data, json_data
from pattern_mining.associat_classifiers import AssociatClassifiers
from net.multimodal import multimodal_utils
from net.multimodal.data_provider.data_tests import test_data_config
import numpy as np
from net.multimodal.data_provider.batch_class import BatchData


json_fname = '../data/fashion53k/json/only_zappos/dataset_dress_all_test.clean.json'
# cnn_fname = '../data/fashion53k/img_regions/4_regions_cnn/per_split/cnn_fc7_test.txt'
cnn_fname = '../data/fashion53k/full_img/per_split/cnn_fc7_test.txt'
imgid2region_indices = multimodal_utils.mk_toy_img_id2region_indices(json_fname,
                                                                     cnn_fname=cnn_fname,
                                                                     num_regions_per_img=1,
                                                                     subset_num_items=-1)

# Create an object of AssociatClassifiers
ac = AssociatClassifiers(json_fname=json_fname,
                         cnn_fname=cnn_fname,
                         img_id2cnn_region_indeces=imgid2region_indices,
                         subset_num_items=20)

# Train a classifier for each img-word pair
ac.fit(option='bernoulli', binarize=0.0,
       subsample=False, verbose=True)

word = 'dress'
y = ac.predict_for_img_id(img_id=6, word=word)

print word, y

word = 'bridesmaid'
y = ac.predict_for_img_id(img_id=6, word=word)
print word, y



# Create a BatchData object
json_fname = test_data_config.exp_config['json_path_test']
cnn_fname = test_data_config.exp_config['cnn_regions_path_test']
imgid2region_indices = multimodal_utils.mk_toy_img_id2region_indices(json_fname, cnn_fname=cnn_fname,
                                                                     num_regions_per_img=5,
                                                                     subset_num_items=-1)
w2v_vocab_fname = test_data_config.exp_config['word2vec_vocab']
w2v_vectors_fname = test_data_config.exp_config['word2vec_vectors']

batch_data = BatchData(json_fname, cnn_fname,
                       imgid2region_indices,
                       w2v_vocab_fname,
                       w2v_vectors_fname,
                       batch_size=2,
                       subset_num_items=2)

batch_data.mk_minibatch(verbose=True, debug=True)

X_img = batch_data.X_img
unique_words_in_batch = batch_data.unique_words_list
# print X_img
print X_img.shape

y = ac.predict_for_cnn(cnn=X_img[7], unique_words_in_batch=unique_words_in_batch)
print y

y = ac.classifiers["dress"].predict(X_img)
print y

# Predict on the batch
y = ac.predict_for_batch(X_img, unique_words_in_batch)
print y



