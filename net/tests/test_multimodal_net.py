"""
Test MultiModalNet against numerical gradient
"""

from net.multimodal import multimodal_net
import numpy as np
import math
from net.gradient_check import eval_numerical_gradient
from net.tests.test_utils import rel_error, mk_random_pair_id
from net.layers import *


####################################################################
# Set loss parameters
####################################################################
loss_params = {}

loss_params['reg'] = reg = 0.

# local loss params
loss_params['use_local'] = use_local = 0.5
loss_params['local_margin'] = local_margin = 1.
loss_params['local_scale'] = local_scale = 1.
loss_params['do_mil'] = do_mil = False

# global loss params
loss_params['use_global'] = use_global = 0.5
loss_params['global_margin'] = global_margin = 40.
loss_params['global_scale'] = global_scale = 1.
loss_params['smooth_num'] = smooth_num = 5.
loss_params['global_method'] = global_method = 'maxaccum'  # 'sum'
loss_params['thrglobalscore'] = thrglobalscore = False

####################################################################
# Create random data
####################################################################
seed = 102
np.random.seed(seed)

N = 9  # number of image-sentence pairs in batch

n_sent_per_img = 5
n_sent = n_sent_per_img * N

n_region_per_img = 3
n_regions = n_region_per_img * N  # number of regions in batch

n_words_per_img = 4
n_words = n_words_per_img * N  # number of words in batch

region2pair_id = mk_random_pair_id(n_region_per_img, N)
word2pair_id = mk_random_pair_id(n_words_per_img, N)

img_input_dim = 13  # size of cnn
txt_input_dim = 11  # size of word2vec pretrained vectors
hidden_dim = 7  # size of multimodal space

std_img = math.sqrt(2. / img_input_dim)
std_txt = math.sqrt(2. / txt_input_dim)
weight_scale = {'img': std_img, 'txt': std_txt}

X_img = np.random.randn(n_regions, img_input_dim)
X_txt = np.random.randn(n_words, txt_input_dim)


####################################################################
# Initialize multimodal net
####################################################################

mmnet = multimodal_net.MultiModalNet(img_input_dim, txt_input_dim, hidden_dim, weight_scale,
                                     use_finetune_cnn=False, use_finetune_w2v=False,
                                     reg=reg, use_local=use_local, use_global=use_global, seed=seed)

mmnet.set_global_score_hyperparams(global_margin=global_margin, global_scale=global_scale,
                                   smooth_num=smooth_num, global_method=global_method,
                                   thrglobalscore=thrglobalscore)

mmnet.set_local_hyperparams(local_margin=local_margin, local_scale=local_scale, do_mil=do_mil)

print 'Testing initialization ... '
Wi2s_std = abs(mmnet.params['Wi2s'].std() - std_img)
# bi2s = mmnet.params['bi2s']
Wsem_std = abs(mmnet.params['Wsem'].std() - std_txt)
bsem = mmnet.params['bsem']
assert Wi2s_std < std_img, 'First layer weights do not seem right'
# assert np.all(bi2s == 0), 'First layer biases do not seem right'
assert Wsem_std < std_txt, 'Second layer weights do not seem right'
# assert np.all(bsem == 0), 'Second layer biases do not seem right'


print 'Testing training loss (no regularization)'
loss, grads = mmnet.loss(X_img, X_txt, region2pair_id, word2pair_id)

print loss
print grads.keys()


print "Testing the gradients"

for reg in [0.0, 0.7, 10, 100, 1000]:
    print 'Running numeric gradient check with reg = ', reg
    mmnet.reg = reg
    loss, grads = mmnet.loss(X_img, X_txt, region2pair_id, word2pair_id)

    for name in sorted(grads):
        f = lambda _: mmnet.loss(X_img, X_txt, region2pair_id, word2pair_id)[0]
        grad_num = eval_numerical_gradient(f, mmnet.params[name], verbose=False)
        print '\n%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))
        print "grad_analaliic\n", grads[name]
        print "grad_num\n", grad_num



