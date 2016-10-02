
import logging
import time
from net.multimodal.data_provider.experiment_data import BatchData, EvaluationData
from net.multimodal.data_provider.cnn_data import CnnData
from net.multimodal.data_provider.word2vec_data import Word2VecData
from net.multimodal import multimodal_net
from net.multimodal.multimodal_solver import MultiModalSolver

from net.multimodal.data_provider.data_tests import test_data_config
from net.multimodal import multimodal_utils
import math

##############################################
# Setup logger
##############################################
fname = test_data_config.exp_config['checkpoint_path'] + '{}_experiment.log.txt'.format(time.strftime('%Y_%m_%d_%H%M'))
logging.basicConfig(filename=fname, level=logging.INFO)

##############################################
# Set batch data object
##############################################

print "setting batch data"
json_fname_train = test_data_config.exp_config['json_path_train']
cnn_fname_train = test_data_config.exp_config['cnn_regions_path_train']
num_regions_per_img = test_data_config.exp_config['num_regions_per_img']
imgid2region_indices_train = multimodal_utils.mk_toy_img_id2region_indices(json_fname_train,
                                                                           num_regions_per_img=num_regions_per_img,
                                                                           subset_num_items=-1)
num_items_train = 20  #len(imgid2region_indices_train)  #TODO: change back
w2v_vocab_fname = test_data_config.exp_config['word2vec_vocab']
w2v_vectors_fname = test_data_config.exp_config['word2vec_vectors']

batch_data = BatchData(json_fname_train, cnn_fname_train,
                       imgid2region_indices_train,
                       w2v_vocab_fname, w2v_vectors_fname,
                       subset_num_items=20)  # TODO: set to -1 on the real experiments


##############################################
# Set evaluation data objects for train and val splits
##############################################
# ______________________________________________
# Train Evaluation Data
# ----------------------------------------------
print "setting evaluation data for train split"
external_vocab_fname = test_data_config.exp_config['external_vocab']
cnn_fname_train = test_data_config.exp_config['cnn_full_img_path_train']
num_regions_per_img = 1
imgid2region_indices_train = multimodal_utils.mk_toy_img_id2region_indices(json_fname_train,
                                                                           num_regions_per_img=1, # For eval data, this is 1
                                                                           subset_num_items=-1)
eval_data_train = EvaluationData(json_fname_train, cnn_fname_train, imgid2region_indices_train,
                                 w2v_vocab_fname, w2v_vectors_fname,
                                 external_vocab_fname, subset_num_items=50) # TODO: set to -1 on the real experiments
# ______________________________________________
# Val Evaluation Data
# ----------------------------------------------
print "setting evaluation data for val split"
json_fname_val = test_data_config.exp_config['json_path_val']
cnn_fname_val = test_data_config.exp_config['cnn_full_img_path_val']
imgid2region_indices_val = multimodal_utils.mk_toy_img_id2region_indices(json_fname_val,
                                                                         num_regions_per_img=1,  # For eval data, this is 1
                                                                         subset_num_items=-1)

eval_data_val = EvaluationData(json_fname_val, cnn_fname_val, imgid2region_indices_val,
                               w2v_vocab_fname, w2v_vectors_fname,
                               external_vocab_fname, subset_num_items=20) # TODO: set to -1 on the real experiments

# ______________________________________________
# Test Evaluation Data
# ----------------------------------------------
print "setting evaluation data for test split"
json_fname_test = test_data_config.exp_config['json_path_test']
cnn_fname_test = test_data_config.exp_config['cnn_full_img_path_test']
imgid2region_indices_test = multimodal_utils.mk_toy_img_id2region_indices(json_fname_test,
                                                                          num_regions_per_img=1, # For eval data, this is 1
                                                                          subset_num_items=-1)

eval_data_test = EvaluationData(json_fname_test, cnn_fname_test, imgid2region_indices_test,
                                w2v_vocab_fname, w2v_vectors_fname,
                                external_vocab_fname, subset_num_items=20) # TODO: set to -1 on the real experiments

# ______________________________________________
# Multi-Word Queries Test Evaluation Data
# ----------------------------------------------
print "setting evaluation data for test split multiple word queries"
json_fname_test = test_data_config.exp_config['json_path_test']
cnn_fname_test = test_data_config.exp_config['cnn_full_img_path_test']
imgid2region_indices_test = multimodal_utils.mk_toy_img_id2region_indices(json_fname_test,
                                                                          num_regions_per_img=1, # For eval data, this is 1
                                                                          subset_num_items=-1)

eval_data_test_mwq = EvaluationData(json_fname_test, cnn_fname_test, imgid2region_indices_test,
                                    w2v_vocab_fname, w2v_vectors_fname,
                                    external_vocab_fname, subset_num_items=20) # TODO: set to -1 on the real experiments



##############################################
# Set the model
##############################################
print "setting the model"
img_input_dim = CnnData(cnn_fname=test_data_config.exp_config['cnn_regions_path_test']).get_cnn_dim()
txt_input_dim = Word2VecData(w2v_vocab_fname, w2v_vectors_fname).get_word2vec_dim()

# hyperparameters
reg = test_data_config.exp_config['reg']
hidden_dim = test_data_config.exp_config['hidden_dim']

# local loss settings
use_local = test_data_config.exp_config['use_local']
local_margin = test_data_config.exp_config['local_margin']
local_scale = test_data_config.exp_config['local_scale']

# global loss settings
# use_global = test_data_config.exp_config['use_global']
# global_margin = test_data_config.exp_config['global_margin']
# global_scale = test_data_config.exp_config['global_scale']
# smooth_num = test_data_config.exp_config['smooth_num']
# global_method = test_data_config.exp_config['global_method']
# thrglobalscore = test_data_config.exp_config['thrglobalscore']

# associat loss settings
use_associat = test_data_config.exp_config['use_associat']

# weight scale for weight initialzation
std_img = math.sqrt(2. / img_input_dim)
std_txt = math.sqrt(2. / txt_input_dim)
weight_scale = {'img': std_img, 'txt': std_txt}

mm_net = multimodal_net.MultiModalNet(img_input_dim, txt_input_dim, hidden_dim, weight_scale,
                                      use_finetune_cnn=False, use_finetune_w2v=False,
                                      reg=reg, use_local=use_local,
                                      use_associat=use_associat, seed=None)
# finetuning starts as false and it can be set to true inside the MultiModalSolver after a number of epochs.

# mm_net.set_global_score_hyperparams(global_margin=global_margin, global_scale=global_scale,
#                                     smooth_num=smooth_num, global_method=global_method,
#                                     thrglobalscore=thrglobalscore)

mm_net.set_local_hyperparams(local_margin=local_margin, local_scale=local_scale, do_mil=False)  # do_mil starts as False and it can be set to True inside MultModalSolver after an number of epochs

##############################################
# Train model with solver
##############################################
solver = MultiModalSolver(mm_net, batch_data,
                          eval_data_train, eval_data_val,
                          eval_data_test, eval_data_test_mwq,
                          num_items_train, test_data_config.exp_config, verbose=True)
# uselocal=uselocal,
# useglobal=useglobal,
# update_rule='sgd',
# optim_config={'learning_rate': lr},
# lr_decay=lr_decay,
# num_epochs=num_epochs,
# batch_size=batch_size,
# print_every=2)

solver.train()


## Later
# for i in xrange(len(lr)):
#     for j in xrange(len(reg)):
#         model = TwoLayerNet(hidden_dim=300, reg=reg[j], weight_scale=1e-2)
#         solver = Solver(model, data,update_rule='rmsprop', optim_config={'learning_rate': lr[i],},
#                       lr_decay=0.95,
#                       num_epochs=10, batch_size=1024,
#                       print_every=1024)
#         solver.train()
#         if solver.best_val_acc > best_val:
#             best_model = model
#             best_val = solver.best_val_acc
#         results[lr[i],reg[j]] = solver.best_val_acc