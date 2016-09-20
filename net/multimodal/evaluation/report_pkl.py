import pickle
from cs231n.multimodal.data_provider.data_tests import test_data_config

# ckpoint_path = test_data_config.exp_config['checkpoint_path']

ckpoint_path = '/Users/susanaparis/Documents/Belgium/DeepFashion/data/fashion53k/experiment_results/'

fname = ckpoint_path + 'report_e_7_m_0_l_0.0_g_1.0_a_0.0_val_f1_0.1021.pkl'

with open(fname, "rb") as f:
    report = pickle.load(f)


fname = ckpoint_path + 'end_report_2016_09_18_0312_tr_0.0603_val_0.0753.pkl'
with open(fname, "rb") as f:
    end_report = pickle.load(f)

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

