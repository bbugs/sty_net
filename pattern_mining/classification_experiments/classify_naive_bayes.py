
from net.multimodal import multimodal_utils as mutils
from pattern_mining import associat_classifiers as asso
from net.multimodal.data_provider.eval_data_class import get_eval_data
import numpy as np
from net.multimodal.evaluation import metrics

from net.multimodal.data_provider.data_tests import test_data_config

exp_config = test_data_config.exp_config

classifiers = asso.get_associat_classifiers(exp_config)
eval_data_train, eval_data_val, eval_data_test, eval_data_test_mwq = get_eval_data(exp_config)

X_img = eval_data_test.X_img
print X_img.shape
print eval_data_test.true_words_list


unique_words = sorted(list(classifiers.word2img_ids_index.keys()))
print unique_words

n_imgs = X_img.shape[0]
n_unique_words = len(unique_words)


img_word_scores = np.zeros((n_imgs, n_unique_words))
for i in range(n_imgs):
    for j in range(n_unique_words):
        p = classifiers.classifiers[unique_words[j]].predict_proba(X_img[i, :].reshape(1, -1))
        img_word_scores[i, j] = p[0][1]

word_ids_pred = np.argsort(-img_word_scores, axis=1).tolist()
true_word_ids = eval_data_test.true_word_ids_list

performance = metrics.avg_prec_recall_all_ks(true_word_ids, word_ids_pred,
                                             Ks=[1, 5, 10, 20])

# print img_word_scores
# print img_word_scores.shape

print "img2txt"
print "P@5", performance['P'][5]
print "P@10", performance['P'][10]
print "P@20", performance['P'][20]

print "\n"
print "R@5", performance['R'][5]
print "R@10", performance['R'][10]
print "R@20", performance['R'][20]

# R@5 0.0188915639916
# R@10 0.0385131798757
# R@20 0.0739857230188


# print classifiers.classifiers["a-line"]
#
# classifiers.classifiers["gown"].predict_proba(X_img[0,:])

# exp_config = {}
#
# rpath = '../data/fashion53k/'
# json_fname = rpath + 'json/only_zappos/dataset_dress_all_test.clean.json'
# cnn_fname = rpath + '/full_img/per_split/cnn_fc7_test.txt'
# subset_train = 20  # normally -1
# num_regions_per_img = 1
# classifier_option = 'bernoulli'
# binarize = 0.0
# subsample = True
#
# imgid2region_indices = mutils.mk_toy_img_id2region_indices(json_fname=json_fname,
#                                                            cnn_fname=cnn_fname,
#                                                            num_regions_per_img=num_regions_per_img,
#                                                            subset_num_items=-1)
#
# #########################################
# # Fit one binary classifier for each word
# #########################################
# associat_classifiers = asso.AssociatClassifiers(json_fname=json_fname,
#                                                 cnn_fname=cnn_fname,
#                                                 img_id2cnn_region_indeces=imgid2region_indices,
#                                                 subset_num_items=subset_train)
#
#
# associat_classifiers.fit(option=classifier_option, binarize=binarize,
#                          subsample=subsample, verbose=True)
#
# #########################################
# # Set up evaluation data
# #########################################
# json_fname_test = exp_config['json_path_test']
# cnn_fname_test = exp_config['cnn_full_img_path_test']
# imgid2region_indices_test = mutils.mk_toy_img_id2region_indices(json_fname_test,
#                                                                           cnn_fname=cnn_fname_test,
#                                                                           num_regions_per_img=num_regions_per_img,
#                                                                           subset_num_items=-1)
#
# eval_data_test = EvaluationData(json_fname_test, cnn_fname_test, imgid2region_indices_test,
#                                 w2v_vocab_fname, w2v_vectors_fname,
#                                 external_vocab_fname,
#                                 subset_num_items=eval_subset_test)

