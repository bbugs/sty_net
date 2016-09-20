from net.multimodal.data_provider import word2vec_data
from net.multimodal.data_provider.data_tests import test_data_config
import numpy as np

d = test_data_config.exp_config

dd = word2vec_data.Word2VecData(w2v_vocab_fname=d['word2vec_vocab'], w2v_vectors_fname=d['word2vec_vectors'])

# X_txt_zappos = dd.get_external_word_vectors()
# print X_txt_zappos

# raw_input("enter key to continue")

# X_txt_zappos = dd.get_word_vectors(external_vocab=True)
# print X_txt_zappos

# raw_input("enter key to continue")

X_txt = dd.get_word_vectors_of_word_list(['random_stuff', 'cat', 'is', 'nice', 'cat'])

# print X_txt
# print X_txt.shape

X_txt = dd.get_word_vectors_of_word_list(['v-neck', 'a-line', 'random-stuff'])  # 264906, 307259

correct_vneck = np.array([-0.11086, -1.006, 0.11159, 0.23023, 0.29037, 0.33901, 0.62025, 0.23046, 0.043336])
correct_aline = np.array([-0.019347, -0.60523, 0.078378, 0.20013, 0.20616, -0.12931, 0.30628, 0.025562, 0.024776])

print X_txt[0, 0:9]
print X_txt[1, 0:9]

assert np.allclose(X_txt[0, 0:9], correct_vneck)
assert np.allclose(X_txt[1, 0:9], correct_aline)

# print "\n\n"
# print X_txt[2,:]  # check the word random-stuff
# print X_txt.shape