from net.multimodal.evaluation.report import Report
from net.multimodal.evaluation.report_unimodal import ReportUnimodal
import json

ckpoint_path = '/Users/susanaparis/Documents/Belgium/Chapter4/data/fashion53k/promising_reports/paris/'

# fname = ckpoint_path + 'report_valf1_0.1214_id_91_hd_1000_l_0.5_g_0.0_a_0.5_e_0_p_0.2109_r_0.3214.pkl'

fname = ckpoint_path +'report_valf1_0.1166_id_46_hd_800_l_1.0_g_0.0_a_0.0_e_0_p_0.2057_r_0.3160.pkl'

# ckpoint_path = '../data/fashion53k/promising_reports/best_all/'
# fname = ckpoint_path + 'report_valf1_9.63_id_15_hd_800_l_1.0_g_0.0_a_0.0_e_2_p_16.82_r_25.26_p_4.74_r_1.47_r_1.70_it_320_p_16.15_r_26.01_p_4.32_r_0.40.pkl'
# fname = ckpoint_path + 'report_valf1_7.91_id_64_hd_1200_l_0.5_g_0.0_a_0.5_e_8_p_14.29_r_18.69_p_6.91_r_2.80_r_1.40_it_1540_p_13.37_r_18.49_p_5.99_r_0.99.pkl'
unimodal = False

# ckpoint_path = '../data/fashion53k/unimodal_exp_results/'
# fname = ckpoint_path + 'report_valf1_6.38_id_3_hd_800_l_0.0_g_0.0_a_0.0_e_1_p_10.27_r_17.37_p_2.91_r_1.43_r_0.00_it_20_p_10.31_r_18.58_p_2.29_r_0.71.pkl'
# unimodal = True

exp_config = {}  # override the stuff that needs to be overridden
exp_config['external_vocab'] = '../data/fashion53k/data_bundles/data_bundle_100/zappos.w2v.vocab.txt'
exp_config['word2vec_vocab'] = '../data/fashion53k/data_bundles/data_bundle_100/w2v.vocab.txt'
exp_config['word2vec_vectors'] = '../data/fashion53k/data_bundles/data_bundle_100/w2v.vecs.txt'
exp_config['json_path_test'] = '../data/fashion53k/data_bundles/data_bundle_100/test.json'
exp_config['json_path_val'] = '../data/fashion53k/data_bundles/data_bundle_100/val.json'

if unimodal:
    r = ReportUnimodal(fname=fname)
else:
    r = Report(fname=fname, exp_config=exp_config)


if True:  # for katrien weights
    import scipy.io as sio
    k_weights = sio.loadmat('../katrien_stuff/Data_v2/weights_only_zappos_words.mat')
    r.mm_net.params['Wi2s'] = k_weights['Wi2s'][:, 0:4096].T  # (1000, 4097)
    r.mm_net.params['bi2s'] = k_weights['Wi2s'][:, -1]  # bi2s is the last column of Wi2s
    r.mm_net.params['Wsem'] = k_weights['Wsem'][:, 0:200].T  # (1000, 201)
    r.mm_net.params['bsem'] = k_weights['Wsem'][:, -1]  # bsem is the last column of Wsem



jf = r.eval_data_test.json_file

true_words, words_pred = r.predict_words(k=20)

idx = 0
data = {}
data['items'] = []

for item in jf.dataset_items:
    item_copy = dict(item)
    item_copy['words_predicted'] = " ".join(words_pred[idx])
    item_copy['words_true'] = " ".join(true_words[idx])
    data['items'].append(item_copy)
    idx += 1

json.dump(data, open('../data/fashion53k/result_plots/words_pred_katrien.json', 'w'),
          indent=4, sort_keys=True)

print "here"

