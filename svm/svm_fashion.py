"""

svm multilabel classifier for our fashion data
"""


from net.multimodal.data_provider.json_data import JsonFile
from net.multimodal.data_provider.cnn_data import CnnData
from net.multimodal.multimodal_utils import check_num_regions
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from net.multimodal.evaluation import metrics

import numpy as np

def mk_y_for_multilabel_classif(json_file):
    vocab = json_file.get_vocab_words_from_json(remove_stops=True, min_word_freq=5)
    # print vocab
    img_ids = json_file.get_img_ids()
    # print img_ids

    y = np.zeros((len(img_ids), len(vocab)), dtype=int)

    i = 0
    for img_id in img_ids:
        # print img_id
        words = json_file.get_word_list_of_img_id(img_id=img_id, remove_stops=True)

        for word in words:
            # print word
            if word not in vocab:
                # print word, "not in vocab"
                continue
            word_idx = vocab.index(word)
            y[i, word_idx] = 1

        i += 1

    return y


def get_X_y(split='test', n_imgs=50):
    # Get X (cnn features):
    cnn_fname = '../data/fashion53k/full_img/per_split//cnn_fc7_{}.txt'.format(split)
    json_fname = '../data/fashion53k/json/only_zappos/dataset_dress_all_{}.clean.json'.format(split)

    json_file = JsonFile(json_fname=json_fname, num_items=n_imgs)
    cnn_data = CnnData(cnn_fname=cnn_fname)

    X = cnn_data.get_cnn()

    if n_imgs > 0:
        X = X[0:n_imgs, :]

    y = mk_y_for_multilabel_classif(json_file=json_file)

    return X, y


print "loading training data"
X_train, y_train = get_X_y(split='train', n_imgs=-1)

print "training classifier"
classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(X_train, y_train)

print "loading test data"
X_test, y_test = get_X_y(split='test', n_imgs=-1)
y_test = y_test.tolist()

print "predicting on test datat"
y_pred = classif.predict(X_test).tolist()

performance_test = metrics.avg_prec_recall_all_ks(ytrue_list=y_test, ypred_list=y_pred, Ks=[1, 5, 10, 20])
print performance_test

# print y.shape
# print X

# print img_ids
# print vocab
# print len(vocab)
