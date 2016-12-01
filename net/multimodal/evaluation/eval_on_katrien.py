import pickle
from net.multimodal.data_provider.data_tests import test_data_config
import numpy as np
import matplotlib.pyplot as plt
from net.multimodal.multimodal_solver import MultiModalSolver
from net.multimodal.data_provider.eval_data_class import EvaluationData, EvaluationDataMWQ
from net.multimodal import multimodal_utils as mm_utils
from net.multimodal import experiment
from net.multimodal.evaluation import metrics
import json


# Load Katrien's weights
import scipy.io as sio
k_weights = sio.loadmat('../katrien_stuff/Data_v2/weights_only_zappos_words.mat')
Wi2s = k_weights['Wi2s'][:, 0:4096].T  # (1000, 4097)
bi2s = k_weights['Wi2s'][:, -1]
Wsem = k_weights['Wsem'][:, 0:200].T  # (1000, 201)
bsem = k_weights['Wsem'][:, -1]

# Load a report just to have the main required stuff
ckpoint_path = '/Users/susanaparis/Documents/Belgium/Chapter4/data/fashion53k/promising_reports/paris/'
fname = ckpoint_path + 'report_valf1_0.1166_id_46_hd_800_l_1.0_g_0.0_a_0.0_e_0_p_0.2057_r_0.3160.pkl'


with open(fname, "rb") as f:
    report = pickle.load(f)

print report['model'].keys()

exp_config = report['exp_config']
exp_config['Ks'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000]
exp_config['eval_k'] = 10
exp_config['associat_margin'] = 1.  # not in report
exp_config['ck_perform_every'] = 20


exp_config['external_vocab'] = '../data/fashion53k/data_bundles/data_bundle_100/zappos.w2v.vocab.txt'
exp_config['word2vec_vocab'] = '../data/fashion53k/data_bundles/data_bundle_100/w2v.vocab.txt'
exp_config['word2vec_vectors'] = '../data/fashion53k/data_bundles/data_bundle_100/w2v.vecs.txt'
exp_config['json_path_test'] = '../data/fashion53k/data_bundles/data_bundle_100/test.json'
exp_config['json_path_val'] = '../data/fashion53k/data_bundles/data_bundle_100/val.json'

#############################################
# Set eval data for TEST
#############################################
print "setting evaluation data for test set"
imgid2region_indices_test = mm_utils.mk_toy_img_id2region_indices(json_fname=exp_config['json_path_test'],
                                                                  cnn_fname=exp_config['cnn_full_img_path_test'],
                                                                  num_regions_per_img=1, subset_num_items=-1)
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
                                                                 cnn_fname=exp_config['cnn_full_img_path_val'],
                                                                 num_regions_per_img=1, subset_num_items=-1)
eval_data_val = EvaluationData(json_fname=exp_config['json_path_val'],
                               cnn_fname=exp_config['cnn_full_img_path_val'],
                               img_id2cnn_region_indeces=imgid2region_indices_val,
                               w2v_vocab_fname=exp_config['word2vec_vocab'],
                               w2v_vectors_fname=exp_config['word2vec_vectors'],
                               external_vocab_fname=exp_config['external_vocab'],
                               subset_num_items=-1)

#############################################
# Set eval data for TEST MWQ
#############################################
# eval_data_test_mwq = EvaluationDataMWQ(json_fname=exp_config['json_path_test'],
#                                        cnn_fname=exp_config['cnn_full_img_path_test'],
#                                        img_id2cnn_region_indeces=imgid2region_indices_test,
#                                        w2v_vocab_fname=exp_config['word2vec_vocab'],
#                                        w2v_vectors_fname=exp_config['word2vec_vectors'],
#                                        external_vocab_fname=exp_config['external_vocab'],
#                                        mwq_aggregator=exp_config['mwq_aggregator'],
#                                        subset_num_items=-1)


#############################################
# Setting the model
#############################################
print "setting the model"
mm_net = experiment.set_model(exp_config)
# Use weigths from the report instead of random init weights
mm_net.params['Wi2s'] = Wi2s  # put katrien's weights in
mm_net.params['bi2s'] = bi2s
mm_net.params['Wsem'] = Wsem
mm_net.params['bsem'] = bsem

#############################################
# Setting the solver
#############################################
print "setting multimodal solver"
solver = MultiModalSolver(model=mm_net, batch_data=None, eval_data_train=None, eval_data_val=None,
                          eval_data_test=eval_data_test, eval_data_test_mwq=None, num_items_train=48689,
                          exp_config=exp_config)

print "i2t test:"
i2t_performance = solver.ck_perform_ranking_img2txt_all_ks(eval_data_test, Ks=exp_config['Ks'])
for K in exp_config['Ks']:
    print "{} \t {:.4f} \t {:.4f}".format(K, 100*i2t_performance['P'][K], 100*i2t_performance['R'][K])

raw_input("press key to continue")


print "t2i test:"
t2i_performance = solver.ck_perform_ranking_txt2img_all_ks(eval_data_test, Ks=exp_config['Ks'])
for K in exp_config['Ks']:
    print "{} \t {:.4f} \t {:.4f}".format(K, 100*t2i_performance['P'][K], 100*t2i_performance['R'][K])





# print "i2t val:"
# print solver.ck_perform_ranking_img2txt_all_ks(eval_data_val, Ks=exp_config['Ks'])


# t2i_mwq_performance = solver.ck_perform_ranking_txt2img_all_ks(eval_data_test_mwq, Ks=exp_config['Ks'])
# print "t2i test mwq:"
# print t2i_mwq_performance

# eval_data_test_mwq = EvaluationDataMWQ(json_fname_test, cnn_fname_test, imgid2region_indices_test,
#                                        w2v_vocab_fname, w2v_vectors_fname,
#                                        external_vocab_fname,
#                                        subset_num_items=-1)


# mm_net = MultiModalNet( img_input_dim, txt_input_dim, hidden_dim, weight_scale,
#                  use_finetune_cnn, use_finetune_w2v,
#                  reg=0.0, use_local=0., use_global=0., use_associat=0.)


# eval_data_train, eval_data_val, eval_data_test, eval_data_test_mwq = get_eval_data(exp_config)


# Plot loss

if False:
    loss = report['loss_history']
    print len(report['loss_history'])
    x = [i for i in range(len(loss))]
    line, = plt.plot(x, loss, 'x', linewidth=2)
    # plt.show()  # uncomment to see plot

    perform_history = report['performance_history']

    # fname = ckpoint_path + 'end_report_2016_09_18_0312_tr_0.0603_val_0.0753.pkl'
    # with open(fname, "rb") as f:
    #     end_report = pickle.load(f)

    # end_report.keys()
    # Out[3]: ['img2txt', 'loss_history', 'txt2img', 'iter', 'exp_config', 'epoch', 'model']

    test_p = []
    for item in report['performance_history']:
        p = item[1]['ranking']['i2t']['test']['R'][10]
        test_p.append(p)

    x = [i for i in range(len(test_p))]
    plt.plot(x, test_p, '-')
    # plt.show()

    test_p = []
    for item in report['performance_history']:
        p = item[1]['ranking']['i2t']['val']['R'][10]
        test_p.append(p)

    x = [i for i in range(len(test_p))]
    plt.plot(x, test_p, '-')
    # plt.show()

    test_p = []
    for item in report['performance_history']:
        p = item[1]['ranking']['i2t']['train']['R'][10]
        test_p.append(p)

    x = [i for i in range(len(test_p))]
    plt.plot(x, test_p, '-')
    # plt.show()


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
