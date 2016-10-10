from net.multimodal.data_provider.batch_class import BatchData

from net.multimodal.data_provider.data_tests import test_data_config
from net.multimodal import multimodal_utils
import numpy as np

json_fname = test_data_config.exp_config['json_path_test']
cnn_fname = test_data_config.exp_config['cnn_regions_path_test']
imgid2region_indices = multimodal_utils.mk_toy_img_id2region_indices(json_fname, cnn_fname=cnn_fname,
                                                                     num_regions_per_img=5,
                                                                     subset_num_items=-1)
w2v_vocab_fname = test_data_config.exp_config['word2vec_vocab']
w2v_vectors_fname = test_data_config.exp_config['word2vec_vectors']

batch_data = BatchData(json_fname, cnn_fname, imgid2region_indices,
                       w2v_vocab_fname, w2v_vectors_fname,
                       batch_size=2,
                       subset_num_items=2)

batch_data.mk_minibatch(verbose=True, debug=True)

# test n_unique_words
assert batch_data.n_unique_words == 11

# test unique_words_list
print batch_data.unique_words_list
assert batch_data.unique_words_list ==['black', 'bridesmaid',
                                      'cocktail', 'dress',
                                      'lace', 'one-shoulder',
                                      'satin', 'sheath',
                                      'short', 'sleeveless',
                                      'zipper']

# test word_seq
assert batch_data.word_seq == ['bridesmaid', 'dress',
                               'black', 'bridesmaid',
                               'cocktail', 'dress',
                               'lace', 'one-shoulder',
                               'satin', 'sheath',
                               'short', 'sleeveless',
                               'zipper']

# test n_regions
assert batch_data.n_regions == 10

# test n_imgs
assert batch_data.n_imgs == 2


################################
# Test y
################################
y = -np.ones((10, 11))
y[0:5, 1] = 1
y[0:5, 3] = 1
y[5:10, :] = 1

print batch_data.y
print y

assert np.array_equal(batch_data.y, y)


################################
# Test X_img
################################

cnn_img1_img2 = np.loadtxt('../data/fashion53k/img_regions/4_regions_cnn/cnn_fc7_img1_img2.txt', delimiter=',')

assert np.array_equal(batch_data.X_img, cnn_img1_img2)

################################
# Test X_txt
################################
assert batch_data.X_txt.shape == (11, 200)
w2v_black = np.loadtxt('../data/word_vects/test/black.txt', delimiter=' ')
print batch_data.X_txt[0,:].shape
print w2v_black.shape

assert np.array_equal(batch_data.X_txt[0,:], w2v_black)


w2v_bridesmaid = np.loadtxt('../data/word_vects/test/bridesmaid.txt', delimiter=' ')
assert np.array_equal(batch_data.X_txt[1,:], w2v_bridesmaid)

w2v_cocktail = np.loadtxt('../data/word_vects/test/cocktail.txt', delimiter=' ')
assert np.array_equal(batch_data.X_txt[2,:], w2v_cocktail)

w2v_dress = np.loadtxt('../data/word_vects/test/dress.txt', delimiter=' ')
assert np.array_equal(batch_data.X_txt[3,:], w2v_dress)

w2v_lace = np.loadtxt('../data/word_vects/test/lace.txt', delimiter=' ')
assert np.array_equal(batch_data.X_txt[4,:], w2v_lace)

w2v_the = np.loadtxt('../data/word_vects/test/the.txt', delimiter=' ')
assert np.array_equal(batch_data.X_txt[5,:], w2v_the)
