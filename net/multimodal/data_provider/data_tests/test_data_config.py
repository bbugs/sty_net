# Set data config dc

exp_config = {}

# assume all data has been precomputed in dc['root_path']
exp_config['root_path'] = root_path = '../'  # assume module is run from sty_net
exp_config['id'] = 0
# exp_config['subset_train'] = 30  # number of items from the train set to use. Normally -1 for experiments

# Image CNN (Full Image + Regions) features
exp_config['num_regions_per_img'] = 4 + 1
exp_config['cnn_regions_path'] = root_path + 'data/fashion53k/img_regions/4_regions_cnn/per_split/'
exp_config['cnn_regions_path_train'] = exp_config['cnn_regions_path'] + '/cnn_fc7_train.txt'
exp_config['cnn_regions_path_val'] = exp_config['cnn_regions_path'] + '/cnn_fc7_val.txt'
exp_config['cnn_regions_path_test'] = exp_config['cnn_regions_path'] + '/cnn_fc7_test.txt'

# Image CNN (Full Image only) features
exp_config['cnn_full_img_path'] = root_path + '/data/fashion53k/full_img/per_split/'
exp_config['cnn_full_img_path_train'] = exp_config['cnn_full_img_path'] + '/cnn_fc7_train.txt'
exp_config['cnn_full_img_path_val'] = exp_config['cnn_full_img_path'] + '/cnn_fc7_val.txt'
exp_config['cnn_full_img_path_test'] = exp_config['cnn_full_img_path'] + '/cnn_fc7_test.txt'

# Text features
exp_config['json_path'] = root_path + 'data/fashion53k/json/only_zappos/'
exp_config['json_path_train'] = exp_config['json_path'] + 'dataset_dress_all_train.clean.json'
exp_config['json_path_val'] = exp_config['json_path'] + 'dataset_dress_all_val.clean.json'
exp_config['json_path_test'] = exp_config['json_path'] + 'dataset_dress_all_test.clean.json'

# Word2vec vectors and vocab
exp_config['word2vec_vocab'] = root_path + 'data/word_vects/glove/vocab.txt'
exp_config['word2vec_vectors'] = root_path + 'data/word_vects/glove/vocab_vecs.txt'

# External vocabulary
exp_config['external_vocab'] = root_path + 'data/fashion53k/external_vocab/zappos.vocab.txt'

# target vocab (used in alignment_data.py on make_y_true_img2txt)
# exp_config['train_vocab'] = root_path + '/fashion53k/vocab_per_split/vocab_train_min_freq_5.txt'  #
# exp_config['val_vocab'] = root_path + '/fashion53k/vocab_per_split/vocab_val_min_freq_5.txt'  # do we need this ??
# exp_config['test_vocab'] = root_path + '/fashion53k/vocab_per_split/vocab_test_min_freq_5.txt'  # do we need this ??

# path to save checkpoints and reports
exp_config['checkpoint_path'] = root_path + '/data/fashion53k/sandbox_results/'

####################################################################
# Set evaluation parameters
####################################################################
exp_config['Ks'] = {1, 5, 10, 15, 20, 30, 75, 100}
exp_config['eval_k'] = 5
exp_config['mwq_aggregator'] = 'avg'

####################################################################
# Set loss parameters
####################################################################
exp_config['use_finetune_cnn'] = False
exp_config['use_finetune_w2v'] = False

exp_config['start_modulation'] = 0.75  # start after 0.75 of epochs

# local loss params
exp_config['use_local'] = 1.
exp_config['local_margin'] = 1.  # keep constant
exp_config['local_scale'] = 1.  # keep constant - regulate with use_local (kept for compatiblity with matlab code)
exp_config['use_mil'] = False

# global loss params
exp_config['use_global'] = 0.
# exp_config['global_margin'] = 40.
# exp_config['global_scale'] = 1.  # keep constant - regulate with use_global (kept for compatiblity with matlab code)
# exp_config['smooth_num'] = 5.
# exp_config['global_method'] = 'maxaccum'  # 'sum'
# exp_config['thrglobalscore'] = False

# association loss params
exp_config['use_associat'] = 0.
exp_config['classifier_type'] = 'naive_bayes'
exp_config['classifier_option'] = 'bernoulli'
exp_config['binarize'] = 0.0
exp_config['classifier_subsample'] = True
exp_config['associat_margin'] = 1.

####################################################################
# Set optimization parameters
####################################################################
exp_config['update_rule'] = 'sgd'

exp_config['reg'] = 0  # regularization
exp_config['hidden_dim'] = 100  # size of multimodal space

optim_config={'learning_rate': 1e-7}
exp_config['optim_config'] = optim_config

exp_config['lr_decay'] = 1

exp_config['print_every'] = 10  # print loss
exp_config['num_epochs'] = 50
exp_config['batch_size'] = 5

exp_config['subset_batch_data'] = 20

exp_config['num_items_train'] = 20  # goes to MultiModalSolver for convenience
exp_config['eval_subset_train'] = 20
exp_config['eval_subset_val'] = 20
exp_config['eval_subset_test'] = 20


