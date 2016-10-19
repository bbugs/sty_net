

import logging
import time
from net.multimodal.data_provider.eval_data_class import get_eval_data
from net.multimodal.data_provider.batch_class import get_batch_data
from net.multimodal.multimodal_solver import MultiModalSolver
from net.unimodal import unimodal_net
import argparse
import json
import os
import random

HIDDEN_DIMS = [100, 200, 500, 800, 1000]


def run_experiment(exp_config, num_items_train,
                   batch_data,
                   eval_data_train, eval_data_val,
                   eval_data_test, eval_data_test_mwq):
    logging.info("id_{} setting model".format(exp_config['id']))
    mm_net = unimodal_net.set_model(exp_config)

    solver = MultiModalSolver(mm_net, batch_data, eval_data_train, eval_data_val, eval_data_test, eval_data_test_mwq,
                              num_items_train, exp_config, unimodal=True, verbose=True)
    print "starting to train id {}".format(exp_config['id'])
    solver.train()

    return solver.status, solver.best_epoch, \
           solver.best_val_f1_score, solver.train_f1_of_best_val


def main(main_args):

    exp_config = vars(main_args)  # convert to ordinary dict
    print json.dumps(exp_config, sort_keys=True, indent=4)

    ##############################################
    # Create database with all conditions
    ##############################################
    print "creating directory to store experiment results"
    # create directory to store database, and reports
    dir_name = exp_config['checkpoint_path'] + '/{}/'.format(time.strftime('%Y_%m_%d_%H%M'))
    exp_config['checkpoint_path'] = dir_name
    os.mkdir(dir_name)

    # a hack
    exp_config['cnn_regions_path_train'] = exp_config['cnn_full_img_path_train']
    exp_config['use_associat'] = 0

    ##############################################
    # Setup logger
    ##############################################
    fname = dir_name + '{}_experiment.log.txt'.format(time.strftime('%Y_%m_%d_%H%M'))
    logging.basicConfig(filename=fname, level=logging.INFO)

    # Build constant data
    print "building eval data"

    eval_data_train, eval_data_val, eval_data_test, eval_data_test_mwq = get_eval_data(exp_config)
    print "finished building eval data"
    num_items_train = exp_config['num_items_train']

    exp_config['use_local'] = 0.
    exp_config['use_global'] = 0.
    exp_config['use_associat'] = 0.

    exp_config['use_mil'] = False
    exp_config['use_finetune_cnn'] = False
    exp_config['use_finetune_w2v'] = False
    exp_config['update_rule'] = 'sgd'

    exp_config['optim_config'] = {'learning_rate': exp_config['learning_rate']}

    ##############################################
    # Run experiment for each condition on the database
    ##############################################

    batch_data = get_batch_data(exp_config)

    for i in range(exp_config['num_exps']):
        exp_config['reg'] = 10 ** random.uniform(-10, -6)  # regularization
        exp_config['learning_rate'] = 10 ** random.uniform(-8, -2)  # learning rate

        exp_config['hidden_dim'] = random.sample(HIDDEN_DIMS, 1)[0]  # choose one

        exp_config['id'] = i
        msg = "Experiment {} out of {}".format(i+1, exp_config['num_exps'])
        logging.info(msg)
        print msg
        print json.dumps(exp_config, indent=4, sort_keys=True)
        logging.info(json.dumps(exp_config, indent=2, sort_keys=True))
        start = time.time()
        status, best_epoch, best_val_f1, train_f1_of_best_val = run_experiment(exp_config=exp_config,
                                                                               num_items_train=num_items_train,
                                                                               batch_data=batch_data,
                                                                               eval_data_train=eval_data_train,
                                                                               eval_data_val=eval_data_val,
                                                                               eval_data_test=eval_data_test,
                                                                               eval_data_test_mwq=eval_data_test_mwq)
        end = time.time()
        time_elapsed = end - start
        msg = "id_{} FINISHED. \t ".format(exp_config['id'])
        msg += "Time: {} \t. ".format(time_elapsed)
        msg += "Status: {} \t. ".format(status)
        msg += "Best epoch: {} \t.".format(best_epoch)
        msg += "Best val_f1: {} \t.".format(best_val_f1)
        msg += "train_f1_of_best_val : {} \t.".format(train_f1_of_best_val)
        logging.info(msg)
        print msg

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # number of image regions
    parser.add_argument('--num_regions_per_img', dest='num_regions_per_img', type=int, default=1)

    # evaluation parameters
    parser.add_argument('--Ks', dest='Ks', default=[1, 5, 10, 15, 20, 30, 50, 75, 100])
    parser.add_argument('--eval_k', dest='eval_k', type=int, default=10)  # eval P@K, R@K to assess best performance
    parser.add_argument('--mwq_aggregator', type=str, default='max')  # or max

    # Image CNN (Full Image only) features
    parser.add_argument('--cnn_full_img_path', dest='cnn_full_img_path', type=str,
                        default='../data/fashion53k/full_img/per_split/')

    parser.add_argument('--cnn_full_img_path_train', dest='cnn_full_img_path_train', type=str,
                        default='../data/fashion53k/full_img/per_split/cnn_fc7_train.txt')

    parser.add_argument('--cnn_full_img_path_val', dest='cnn_full_img_path_val', type=str,
                        default='../data/fashion53k/full_img/per_split/cnn_fc7_val.txt')

    parser.add_argument('--cnn_full_img_path_test', dest='cnn_full_img_path_test', type=str,
                        default='../data/fashion53k/full_img/per_split/cnn_fc7_test.txt')

    # Json files with text
    parser.add_argument('--json_path', dest='json_path', type=str,
                        default='../data/fashion53k/json/only_zappos/')

    parser.add_argument('--json_path_train', dest='json_path_train', type=str,
                        default='../data/fashion53k/json/only_zappos/dataset_dress_all_train.clean.json')

    parser.add_argument('--json_path_val', dest='json_path_val', type=str,
                        default='../data/fashion53k/json/only_zappos/dataset_dress_all_val.clean.json')

    parser.add_argument('--json_path_test', dest='json_path_test', type=str,
                        default='../data/fashion53k/json/only_zappos/dataset_dress_all_test.clean.json')

    # Word2vec vectors and vocab
    parser.add_argument('--word2vec_vocab', dest='word2vec_vocab', type=str,
                        default='../data/word_vects/glove/vocab.txt')
    parser.add_argument('--word2vec_vectors', dest='word2vec_vectors', type=str,
                        default='../data/word_vects/glove/vocab_vecs.txt')

    # External vocabulary
    parser.add_argument('--external_vocab', dest='external_vocab', type=str,
                        default='../data/fashion53k/external_vocab/zappos.vocab.txt')

    # target vocab (used in alignment_data.py on make_y_true_img2txt)
    # TODO: see where min_freq applies (perhaps make the clean json already remove words with less than min_freq)
    # parser.add_argument('--train_vocab', dest='train_vocab', type=str,
    #                     default='../data/fashion53k/vocab_per_split/vocab_train_min_freq_5.txt')
    #
    # parser.add_argument('--val_vocab', dest='val_vocab', type=str,
    #                     default='../data/fashion53k/vocab_per_split/vocab_val_min_freq_5.txt')
    #
    # parser.add_argument('--test_vocab', dest='test_vocab', type=str,
    #                     default='../data/fashion53k/vocab_per_split/vocab_test_min_freq_5.txt')

    # path to save checkpoints and reports
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', type=str,
                        default='../data/fashion53k/unimodal_exp_results/')

    # other params fixed
    # start after 0.75 of epochs
    parser.add_argument('--start_modulation', dest='start_modulation', type=float, default=0.75)
    parser.add_argument('--print_every', dest='print_every', type=int, default=10)  # print loss
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=3)  # number of epochs
    parser.add_argument('--update_rule', dest='update_rule', type=str, default='sgd')  # update rule
    parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=0.90)  # learning rate decay

    # number of experiments
    parser.add_argument('--num_exps', dest='num_exps', type=int, default=100)  # number of conditions on database

    # number of items for train, val and test
    parser.add_argument('--num_items_train', dest='num_items_train', type=int, default=211)  # 48,689 or 211 if word driven batch
    #  num_items_train goes to MultiModalSolver for convenience
    parser.add_argument('--eval_subset_train', dest='eval_subset_train', type=int, default=50)  # -1 to use them all
    parser.add_argument('--eval_subset_val', dest='eval_subset_val', type=int, default=10)  # -1 to use them all
    parser.add_argument('--eval_subset_test', dest='eval_subset_test', type=int, default=5)  # -1 to use them all

    # batch data params
    parser.add_argument('--subset_batch_data', dest='subset_batch_data', type=int, default=100)  # -1 to use them all
    # subset_batch_data goes to get_batch_data and get_associat_classifiers
    # which affects BatchData and BatchDataAssociat.  It is not the batch_size.
    # It's just to be able to do quick tests with a subset of the training data

    parser.add_argument('--word_driven_batch', dest='word_driven_batch', type=bool, default=True)
    parser.add_argument('--max_n_imgs_per_word', dest='max_n_imgs_per_word', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1)
    parser.add_argument('--ck_perform_every', dest='ck_perform_every', type=int, default=20)
    # if word_driven_bath is True, batch_size should be <= 1.

    args = parser.parse_args()

    main(args)

