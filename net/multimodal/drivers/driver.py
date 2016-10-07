
import logging
import time
from net.multimodal.data_provider.eval_data_class import get_eval_data
from net.multimodal.data_provider.batch_class import get_batch_data
from net.multimodal.multimodal_solver import MultiModalSolver
from net.multimodal import experiment
from net.multimodal.experiment_db import exp_db_populator
from sqlalchemy.pool import StaticPool
# from net.multimodal import data_config
import argparse
import threading
import Queue
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker
import json
import os

from net.multimodal.experiment_db.experiment_db_setup import Base, Experiment
# https://docs.python.org/3/library/queue.html
# http://docs.sqlalchemy.org/en/latest/dialects/sqlite.html#threading-pooling-behavior


def run_experiment(exp_config, num_items_train,
                   eval_data_train, eval_data_val,
                   eval_data_test, eval_data_test_mwq):
    logging.info("id_{} setting model".format(exp_config['id']))
    mm_net = experiment.set_model(exp_config)

    logging.info("id_{} model has been set. Going to create new batch object".format(exp_config['id']))
    batch_data = get_batch_data(exp_config)

    logging.info("id_{} created new batch object".format(exp_config['id']))
    solver = MultiModalSolver(mm_net, batch_data,
                              eval_data_train, eval_data_val,
                              eval_data_test, eval_data_test_mwq,
                              num_items_train, exp_config, verbose=True)
    print "starting to train id {}".format(exp_config['id'])
    solver.train()

    return solver.status, solver.best_epoch, \
           solver.best_val_f1_score, solver.train_f1_of_best_val


def main(main_args):

    main_config = vars(main_args)  # convert to ordinary dict
    print json.dumps(main_config, sort_keys=True, indent=4)

    ##############################################
    # Create database with all conditions
    ##############################################
    print "creating directory to store experiment database and results"
    # create directory to store database, and reports
    dir_name = main_config['checkpoint_path'] + '/{}/'.format(time.strftime('%Y_%m_%d_%H%M'))
    main_config['checkpoint_path'] = dir_name
    os.mkdir(dir_name)

    db_name = 'sqlite:///' + main_config['checkpoint_path'] + main_config['experiment_db']
    exp_db_populator.populate_db(db_name, n_exps=main_config['num_exps'])

    print "connecting to database"
    engine = create_engine(db_name, echo=False,
                           connect_args={'check_same_thread': False},
                           poolclass=StaticPool)  # make session available to threads

    Base.metadata.bind = engine
    db_session = sessionmaker(bind=engine)
    session = db_session()

    ##############################################
    # Setup logger
    ##############################################
    fname = dir_name + '{}_experiment.log.txt'.format(time.strftime('%Y_%m_%d_%H%M'))
    logging.basicConfig(filename=fname, level=logging.INFO)

    # Build constant data
    print "building eval data"

    eval_data_train, eval_data_val, eval_data_test, eval_data_test_mwq = get_eval_data(main_config)
    print "finished building eval data"
    num_items_train = main_config['num_items_train']

    ##############################################
    # get list of experiment conditions  # get all configs not done sorted by priority
    ##############################################
    exp_configs_list = []
    configs = session.query(Experiment).filter_by(done=False).order_by(desc(Experiment.priority))
    # todo: add status
    for c in configs:
        d = vars(c)  # convert to dictionary
        if '_sa_instance_state' in d.keys():
            del d['_sa_instance_state']  # remove field inhereted by sqlalchemy
        d.update(main_config)  # include global config
        d['optim_config'] = {'learning_rate': d['learning_rate']}
        exp_configs_list.append(d)
        # print d

    ##############################################
    # Run experiment for each condition on the database
    ##############################################

    i = 1
    for exp_config in exp_configs_list:
        msg = "Experiment {} out of {}".format(i, len(exp_configs_list))
        logging.info(msg)
        print msg
        print json.dumps(exp_config, indent=4, sort_keys=True)
        logging.info(json.dumps(exp_config, indent=2, sort_keys=True))
        start = time.time()
        status, best_epoch, best_val_f1, train_f1_of_best_val = run_experiment(exp_config, num_items_train,
                                                                               eval_data_train, eval_data_val,
                                                                               eval_data_test, eval_data_test_mwq)
        end = time.time()
        row = session.query(Experiment).filter_by(id=exp_config['id']).first()
        row.done = True
        row.time = end - start
        row.status = status
        print "status", status
        row.best_val_f1 = best_val_f1
        row.train_f1_of_best_val = train_f1_of_best_val
        row.best_epoch = best_epoch
        # write to database results of experiment
        session.commit()
        print "finished item {}".format(exp_config['id'])
        i += 1

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # number of image regions
    parser.add_argument('--num_regions_per_img', dest='num_regions_per_img', type=int, default=5)

    # evaluation parameters
    parser.add_argument('--Ks', dest='Ks', default=[1, 5, 10, 15, 20, 30, 75, 100])
    parser.add_argument('--eval_k', dest='eval_k', type=int, default=10)  # eval P@K, R@K to assess best performance
    parser.add_argument('--mwq_aggregator', type=str, default='max')  # or max

    # Image CNN (Full Image + Regions) features
    parser.add_argument('--cnn_regions_path', dest='cnn_regions_path', type=str,
                        default='../data/fashion53k/img_regions/4_regions_cnn/per_split/')

    parser.add_argument('--cnn_regions_path_train', dest='cnn_regions_path_train', type=str,
                        default='../data/fashion53k/img_regions/4_regions_cnn/per_split/cnn_fc7_train.txt')

    parser.add_argument('--cnn_regions_path_val', dest='cnn_regions_path_val', type=str,
                        default='../data/fashion53k/img_regions/4_regions_cnn/per_split/cnn_fc7_val.txt')

    parser.add_argument('--cnn_regions_path_test', dest='cnn_regions_path_test', type=str,
                        default='../data/fashion53k/img_regions/4_regions_cnn/per_split/cnn_fc7_test.txt')

    # Image CNN (Full Image only) features
    # TODO: see where num_regions apply
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
                        default='../data/fashion53k/experiment_results/')

    # name of experiment database experiment.db
    parser.add_argument('--experiment_db', dest='experiment_db', type=str,
                        default='experiments.db')

    # local loss params constants
    parser.add_argument('--local_margin', dest='local_margin', type=float, default=1.)
    parser.add_argument('--local_scale', dest='local_scale', type=float, default=1.)  # keep constant - regulate with
    #  use_local (kept for compatiblity with matlab code)

    # global loss params constants
    parser.add_argument('--global_scale', dest='global_scale', type=float, default=1.)  # keep constant - regulate with
    # use_global (kept for compatiblity with matlab code)
    parser.add_argument('--smooth_num', dest='smooth_num', type=float, default=5.)
    parser.add_argument('--global_method', dest='global_method', type=str, default='maxaccum')

    # other params fixed
    # start after 0.75 of epochs
    parser.add_argument('--start_modulation', dest='start_modulation', type=float, default=0.75)
    parser.add_argument('--print_every', dest='print_every', type=int, default=10)  # print loss
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=20)  # number of epochs
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100)  # batch size
    parser.add_argument('--update_rule', dest='update_rule', type=str, default='sgd')  # update rule
    parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=0.90)  # learning rate decay

    # number of threads
    parser.add_argument("-t", dest="num_threads", default=1, help="number of threads")

    # number of experiments
    parser.add_argument('--num_exps', dest='num_exps', type=int, default=100)  # number of conditions on database

    # number of items for train, val and test
    parser.add_argument('--num_items_train', dest='num_items_train', type=int, default=100)  # 48,689
    #  num_items_train goes to MultiModalSolver for convenience
    parser.add_argument('--eval_subset_train', dest='eval_subset_train', type=int, default=50)  # -1 to use them all
    parser.add_argument('--eval_subset_val', dest='eval_subset_val', type=int, default=10)  # -1 to use them all
    parser.add_argument('--eval_subset_test', dest='eval_subset_test', type=int, default=5)  # -1 to use them all

    # subset_batch_data goes to get_batch_data and get_associat_classifiers
    # which affects BatchData and BatchDataAssociat.  It is not the batch_size.
    # It's just to be able to do quick tests with a subset of the training data
    parser.add_argument('--subset_batch_data', dest='subset_batch_data', type=int, default=100)  # -1 to use them all

    # configuration of association classifiers
    parser.add_argument('--classifier_type', dest='classifier_type', type=str, default='naive_bayes')
    parser.add_argument('--classifier_option', dest='classifier_option', type=str, default='bernoulli') #or multinomial
    parser.add_argument('--binarize', dest='binarize', type=float, default=0.0)
    parser.add_argument('--classifier_subsample', dest='classifier_subsample', type=bool, default=True) # subsample pos or negative images so that we have the same number of pos and neg images
    parser.add_argument('--associat_margin', dest='associat_margin', type=float, default=1.)

    args = parser.parse_args()

    main(args)

