import pickle
from net.multimodal.data_provider.data_tests import test_data_config
import numpy as np
import matplotlib.pyplot as plt
from net.multimodal.multimodal_solver import MultiModalSolver
from net.multimodal.data_provider.experiment_data import EvaluationDataMWQ
from net.multimodal.data_provider.eval_data_class import EvaluationData, EvaluationDataMWQ
from net.multimodal import multimodal_utils as mm_utils
from net.multimodal import experiment
from net.multimodal.evaluation import metrics
import json


ckpoint_path = '/Users/susanaparis/Documents/Belgium/Chapter4/data/fashion53k/promising_reports/paris/'
# fname = ckpoint_path + 'report_valf1_0.0420_id_41_hd_800_l_1.0_g_0.0_a_0.0_e_17_p_0.0771_r_0.1017.pkl'
fname = ckpoint_path + 'report_valf1_0.1166_id_41_hd_500_l_1.0_g_0.0_a_0.0_e_0_p_0.2041_r_0.3163.pkl'

with open(fname, "rb") as f:
    report = pickle.load(f)

print report['model'].keys()

exp_config = report['exp_config']
exp_config['Ks'] = [1, 5, 10, 20]
exp_config['eval_k'] = 10


#############################################
# Set eval data for TEST
#############################################
print "setting evaluation data for test set"
imgid2region_indices_test = mm_utils.mk_toy_img_id2region_indices(json_fname=exp_config['json_path_test'],
                                                                  num_regions_per_img=1,
                                                                  subset_num_items=-1)
eval_data_test = EvaluationData(json_fname=exp_config['json_path_test'],
                                cnn_fname=exp_config['cnn_full_img_path_test'],
                                img_id2cnn_region_indeces=imgid2region_indices_test,
                                w2v_vocab_fname=exp_config['word2vec_vocab'],
                                w2v_vectors_fname=exp_config['word2vec_vectors'],
                                external_vocab_fname=exp_config['external_vocab'],
                                subset_num_items=-1)

#############################################
# Set eval data for VAL
#############################################
imgid2region_indices_val = mm_utils.mk_toy_img_id2region_indices(json_fname=exp_config['json_path_val'],
                                                                 num_regions_per_img=1,
                                                                 subset_num_items=-1)
eval_data_val = EvaluationData(json_fname=exp_config['json_path_val'],
                               cnn_fname=exp_config['cnn_full_img_path_val'],
                               img_id2cnn_region_indeces=imgid2region_indices_val,
                               w2v_vocab_fname=exp_config['word2vec_vocab'],
                               w2v_vectors_fname=exp_config['word2vec_vectors'],
                               external_vocab_fname=exp_config['external_vocab'],
                               subset_num_items=-1)

#############################################
# Setting the model
#############################################
print "setting the model"
mm_net = experiment.set_model(exp_config)
# Use weigths from the report instead of random init weights
mm_net.params['Wi2s'] = report['model']['Wi2s']
mm_net.params['bi2s'] = report['model']['bi2s']
mm_net.params['Wsem'] = report['model']['Wsem']
mm_net.params['bsem'] = report['model']['bsem']

#############################################
# Setting the solver
#############################################
print "setting multimodal solver"
solver = MultiModalSolver(model=mm_net, batch_data=None,
                          eval_data_train=None, eval_data_val=None,
                          eval_data_test=eval_data_test,
                          eval_data_test_mwq=None,
                          num_items_train=48689, exp_config=exp_config)

print solver.ck_perform_ranking_img2txt_all_ks(eval_data_test, Ks=exp_config['Ks'])
print solver.ck_perform_ranking_img2txt_all_ks(eval_data_val, Ks=exp_config['Ks'])

t2i_performance = solver.ck_perform_ranking_txt2img_all_ks(eval_data_test, Ks=exp_config['Ks'])
print t2i_performance

# eval_data_test_mwq = EvaluationDataMWQ(json_fname_test, cnn_fname_test, imgid2region_indices_test,
#                                        w2v_vocab_fname, w2v_vectors_fname,
#                                        external_vocab_fname,
#                                        subset_num_items=-1)


# mm_net = MultiModalNet( img_input_dim, txt_input_dim, hidden_dim, weight_scale,
#                  use_finetune_cnn, use_finetune_w2v,
#                  reg=0.0, use_local=0., use_global=0., use_associat=0.)


# eval_data_train, eval_data_val, eval_data_test, eval_data_test_mwq = get_eval_data(exp_config)


# Plot loss
loss = report['loss_history']
print len(report['loss_history'])
x = [i for i in range(len(loss))]
line, = plt.plot(x, loss, 'x', linewidth=2)
# plt.show()  # uncomment to see plot



# fname = ckpoint_path + 'end_report_2016_09_18_0312_tr_0.0603_val_0.0753.pkl'
# with open(fname, "rb") as f:
#     end_report = pickle.load(f)

# end_report.keys()
# Out[3]: ['img2txt', 'loss_history', 'txt2img', 'iter', 'exp_config', 'epoch', 'model']

print "here"

# report.keys()
# Out[4]: ['img2txt', 'loss_history', 'txt2img', 'iter', 'exp_config', 'epoch', 'model']

# report['model'].keys()
# Out[6]: ['bi2s', 'Wsem', 'Wi2s', 'bsem']

# report['img2txt']
# {'train_current_performance': {'f1': 0.049829695975778986,
#   'p': 0.027828659997181909,
#   'r': 0.23795180722891565},
#  'train_history': {'f1': [0.05738763425619705, 0.049829695975778986],
#   'p': [0.030276845358768987, 0.027828659997181909],
#   'r': [0.54879518072289157, 0.23795180722891565]},
#  'val_current_performance': {'f1': 0.078930202217873446,
#   'p': 0.046086459721957719,
#   'r': 0.27468785471055618},
#  'val_history': {'f1': [0.071792384186272137, 0.078930202217873446],
#   'p': [0.038649025069637882, 0.046086459721957719],
#   'r': [0.50397275822928489, 0.27468785471055618]}}

