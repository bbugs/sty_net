"""
Test the MultiModalNet against matlab implementation
susy_code_bare_bones/check_python_cost.m
"""

import os
from cs231n.multimodal import multimodal_net
import numpy as np
import math
from cs231n.layers import *
import scipy.io as sio
from cs231n.tests import test_utils

rpath = '../../nips2014_karpathy/susy_code_bare_bones/'

####################################################################
# Set loss parameters
####################################################################
loss_params = {}

loss_params['reg'] = reg = 0.
loss_params['finetuneCNN'] = False

# local loss params
loss_params['uselocal'] = use_local = False
loss_params['local_margin'] = local_margin = 1.
loss_params['local_scale'] = local_scale = 1.
loss_params['do_mil'] = do_mil = False

# global loss params
loss_params['useglobal'] = use_global = True
loss_params['global_margin'] = global_margin = 40.
loss_params['global_scale'] = global_scale = 1.
loss_params['smooth_num'] = smotth_num = 5.
loss_params['global_method'] = global_method = 'sum'
loss_params['thrglobalscore'] = thrglobalscore = False

sio.savemat(rpath + 'loss_params.mat', {'loss_params': loss_params})

# for my implementation, use_local and use_global are floats
if use_local:
    use_local = 1.
if use_global:
    use_global = 1.

####################################################################
# Create random data
####################################################################
seed = 102
np.random.seed(seed)

N = 100  # number of image-sentence pairs in batch

n_sent_per_img = 5
n_sent = n_sent_per_img * N

n_region_per_img = 3
n_regions = n_region_per_img * N  # number of regions in batch

n_words_per_img = 4
n_words = n_words_per_img * N  # number of words in batch

region2pair_id = test_utils.mk_random_pair_id(n_region_per_img, N)
word2pair_id = test_utils.mk_random_pair_id(n_words_per_img, N)

img_input_dim = 13  # size of cnn
txt_input_dim = 11  # size of word2vec pretrained vectors
hidden_dim = 7  # size of multimodal space

std_img = math.sqrt(2. / img_input_dim)
std_txt = math.sqrt(2. / txt_input_dim)
weight_scale = {'img': std_img, 'txt': std_txt}

X_img = np.random.randn(n_regions, img_input_dim)
X_txt = np.random.randn(n_words, txt_input_dim)

# Replace the last column of X by 1's so that the last row of W is b.
X_img[:, -1] = 1
X_txt[:, -1] = 1



####################################################################
# Initialize multimodal net
####################################################################

mmnet = multimodal_net.MultiModalNet(img_input_dim, txt_input_dim, hidden_dim, weight_scale,
                                     use_finetune_cnn=False, use_finetune_w2v=False,
                                     reg=reg, use_local=use_local, use_global=use_global, seed=seed)

Wi2s = mmnet.params['Wi2s']
Wsem = mmnet.params['Wsem']
# When comparing with matlab, bs should always be zeros in the multimodal net
bi2s = mmnet.params['bi2s'] = np.zeros(hidden_dim)  # set b's to zero so we can compare with matlab
bsem = mmnet.params['bsem'] = np.zeros(hidden_dim)

# The last row of W becomes b
Wi2s[-1, :] = 0  # Set the last row of W to zero
Wsem[-1, :] = 0  # Set the last row of W to zero

sio.savemat(rpath + 'X_img.mat', {'X_img': X_img})
sio.savemat(rpath + 'X_txt.mat', {'X_txt': X_txt})
sio.savemat(rpath + 'Wi2s.mat', {'Wi2s': Wi2s})
sio.savemat(rpath + 'Wsem.mat', {'Wsem': Wsem})
# sio.savemat(rpath + 'bi2s.mat', {'bi2s': bi2s})
# sio.savemat(rpath + 'bsem.mat', {'bsem': bsem})
sio.savemat(rpath + 'region2pair_id.mat', {'region2pair_id': region2pair_id})
sio.savemat(rpath + 'word2pair_id.mat', {'word2pair_id': word2pair_id})


mmnet.set_global_score_hyperparams(global_margin=global_margin, global_scale=global_scale,
                                   smooth_num=smotth_num, global_method=global_method,
                                   thrglobalscore=thrglobalscore)

mmnet.set_local_hyperparams(local_margin=local_margin, local_scale=local_scale, do_mil=do_mil)

loss, grads = mmnet.loss(X_img, X_txt, region2pair_id, word2pair_id)


###################
# Call Matlab check_python_cost.m to compute cost and gradients

os.system("matlab -nojvm -nodesktop < {0}/check_python_cost.m".format(rpath))

# Load matlab output
matlab_output = sio.loadmat(rpath + 'matlab_output.mat')

# Compare the loss
print "\nloss", loss
print "matlab_cost", matlab_output['cost'][0][0]

assert np.allclose(loss, matlab_output['cost'][0][0])

# Compare the bias weights
print "\n\n, bsem \n", grads['bsem']
print "\n\n, bsem matlab \n", matlab_output['df_Wsem'].T[-1, :]

bsem_matlab = matlab_output['df_Wsem'].T[-1, :]
assert np.allclose(grads['bsem'], bsem_matlab)

print "\n\n, bi2s \n", grads['bi2s']
print "\n\n, bi2s matlab \n", matlab_output['df_Wi2s'].T[-1, :]
bi2s_matlab = matlab_output['df_Wi2s'].T[-1, :]
assert np.allclose(grads['bi2s'], bi2s_matlab)

# Compare the weights
print "\n\n\n dWi2s\n", grads['Wi2s'].T, "\n\n\n"
print "matlab df_Wi2s\n", matlab_output['df_Wi2s']

print "\n\n\n dWsem\n", grads['Wsem'].T, "\n\n\n"
print "matlab df_Wsem\n", matlab_output['df_Wsem']

assert np.allclose(grads['Wi2s'].T, matlab_output['df_Wi2s'])
assert np.allclose(grads['Wsem'].T, matlab_output['df_Wsem'])

print test_utils.rel_error(grads['Wi2s'].T, matlab_output['df_Wi2s'])
print test_utils.rel_error(grads['Wsem'].T, matlab_output['df_Wsem'])







