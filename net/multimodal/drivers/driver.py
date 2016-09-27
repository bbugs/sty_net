
import logging
import time
from net.multimodal.data_provider.experiment_data import get_batch_data, get_eval_data
from net.multimodal.multimodal_solver import MultiModalSolver
from net.multimodal import experiment
from sqlalchemy.pool import StaticPool
# from net.multimodal import data_config
import argparse
import threading
import Queue
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker
import json

from net.multimodal.experiment_db.experiment_db_setup import Base, Experiment
# https://docs.python.org/3/library/queue.html
# http://docs.sqlalchemy.org/en/latest/dialects/sqlite.html#threading-pooling-behavior


def run_experiment(exp_config):

    mm_net = experiment.set_model(exp_config)

    solver = MultiModalSolver(mm_net, BATCH_DATA, EVAL_DATA_TRAIN, EVAL_DATA_VAL,
                              NUM_ITEMS_TRAIN, exp_config, verbose=True)
    print "starting to train id {}".format(exp_config['id'])
    solver.train()

    return solver.status, solver.best_epoch, \
           solver.best_val_f1_score, solver.train_f1_of_best_val


def worker():
    while True:
        exp_config = q.get()
        start = time.time()
        status, best_epoch, best_val_f1, train_f1_of_best_val = run_experiment(exp_config)
        end = time.time()
        row = SESSION.query(Experiment).filter_by(id=exp_config['id']).first()
        row.done = True
        row.time = end - start
        row.status = status
        print "status", status
        row.best_val_f1 = best_val_f1
        row.train_f1_of_best_val = train_f1_of_best_val
        row.best_epoch = best_epoch
        SESSION.commit()
        q.task_done()
        # write to database that this item has finished
        print "finished item {}".format(exp_config['id'])


def main(args):

    global_config = vars(args)

    # get list of experiment conditions  # get all configs not done sorted by priority
    exp_configs_list = []
    configs = SESSION.query(Experiment).filter_by(done=False).order_by(desc(Experiment.priority))
    # todo: add status
    for c in configs:
        d = vars(c)  # convert to dictionary
        if '_sa_instance_state' in d.keys():
            del d['_sa_instance_state']  # remove field inhereted by sqlalchemy
        d.update(global_config)  # include global config
        d['optim_config'] = {'learning_rate': d['learning_rate']}
        exp_configs_list.append(d)
        # print d

    # run_experiment()
    global q
    q = Queue.Queue()

    # initialize the worker threads
    threads = []
    for i in range(global_config['num_threads']):
        print "start thread {}".format(i)
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        threads.append(t)

    # put experiments in the queue
    for exp in exp_configs_list:
        q.put(exp)

    q.join()       # block until all tasks are done

    # stop workers
    for i in range(global_config['num_threads']):
        q.put(None)
    for t in threads:
        t.join()
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # root path
    parser.add_argument('--root_path', dest='root_path', type=str, default='../data/')
    args = parser.parse_args()
    root_path = args.root_path

    # number of image regions
    parser.add_argument('--num_regions_per_img', dest='num_regions_per_img', type=int, default=5)

    # Image CNN (Full Image + Regions) features
    parser.add_argument('--cnn_regions_path', dest='cnn_regions_path', type=str,
                        default=root_path + '/fashion53k/img_regions/4_regions_cnn/per_split/')

    parser.add_argument('--cnn_regions_path_train', dest='cnn_regions_path_train', type=str,
                        default=root_path + '/fashion53k/img_regions/4_regions_cnn/per_split/cnn_fc7_train.txt')

    parser.add_argument('--cnn_regions_path_val', dest='cnn_regions_path_val', type=str,
                        default=root_path + '/fashion53k/img_regions/4_regions_cnn/per_split/cnn_fc7_val.txt')

    parser.add_argument('--cnn_regions_path_test', dest='cnn_regions_path_test', type=str,
                        default=root_path + '/fashion53k/img_regions/4_regions_cnn/per_split/cnn_fc7_test.txt')

    # Image CNN (Full Image only) features
    # TODO: Make cnn features for full images only
    # TODO: see where num_regions apply
    parser.add_argument('--cnn_full_img_path', dest='cnn_full_img_path', type=str,
                        default=root_path + '/fashion53k/full_img/per_split/')

    parser.add_argument('--cnn_full_img_path_train', dest='cnn_full_img_path_train', type=str,
                        default=root_path + '/fashion53k/full_img/per_split/cnn_fc7_train.txt')

    parser.add_argument('--cnn_full_img_path_val', dest='cnn_full_img_path_val', type=str,
                        default=root_path + '/fashion53k/full_img/per_split/cnn_fc7_val.txt')

    parser.add_argument('--cnn_full_img_path_test', dest='cnn_full_img_path_test', type=str,
                        default=root_path + '/fashion53k/full_img/per_split/cnn_fc7_test.txt')

    # Json files with text
    parser.add_argument('--json_path', dest='json_path', type=str,
                        default=root_path + '/fashion53k/json/with_ngrams/')

    parser.add_argument('--json_path_train', dest='json_path_train', type=str,
                        default=root_path + '/fashion53k/json/with_ngrams/dataset_dress_all_train.clean.json')

    parser.add_argument('--json_path_val', dest='json_path_val', type=str,
                        default=root_path + '/fashion53k/json/with_ngrams/dataset_dress_all_val.clean.json')

    parser.add_argument('--json_path_test', dest='json_path_test', type=str,
                        default=root_path + '/fashion53k/json/with_ngrams/dataset_dress_all_test.clean.json')

    # Word2vec vectors and vocab
    parser.add_argument('--word2vec_vocab', dest='word2vec_vocab', type=str,
                        default=root_path + '/word_vects/glove/vocab.txt')
    parser.add_argument('--word2vec_vectors', dest='word2vec_vectors', type=str,
                        default=root_path + '/word_vects/glove/vocab_vecs.txt')

    # External vocabulary
    parser.add_argument('--external_vocab', dest='external_vocab', type=str,
                        default=root_path + '/fashion53k/external_vocab/zappos.vocab.txt')

    # target vocab (used in alignment_data.py on make_y_true_img2txt)
    # TODO: see where min_freq applies (perhaps make the clean json already remove words with less than min_freq)
    # parser.add_argument('--train_vocab', dest='train_vocab', type=str,
    #                     default=root_path + '/fashion53k/vocab_per_split/vocab_train_min_freq_5.txt')
    #
    # parser.add_argument('--val_vocab', dest='val_vocab', type=str,
    #                     default=root_path + '/fashion53k/vocab_per_split/vocab_val_min_freq_5.txt')
    #
    # parser.add_argument('--test_vocab', dest='test_vocab', type=str,
    #                     default=root_path + '/fashion53k/vocab_per_split/vocab_test_min_freq_5.txt')

    # path to save checkpoints and reports
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', type=str,
                        default=root_path + '/fashion53k/experiment_results/')

    # path to experiment.db
    parser.add_argument('--experiment_db', dest='experiment_db', type=str,
                        default='experiments.db')

    # todo: check out why results are the same. there might be a seed set somewhere?

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
    parser.add_argument('--print_every', dest='print_every', type=int, default=2)  # print loss
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=7)  # number of epochs
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=5)  # batch size
    parser.add_argument('--update_rule', dest='update_rule', type=str, default='sgd')  # update rule
    parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=0.95)  # learning rate decay

    # number of threads
    parser.add_argument("-t", dest="num_threads", default=4, help="number of threads")

    args = parser.parse_args()

    GLOBAL_CONFIG = vars(args)  # convert to ordinary dict
    print json.dumps(GLOBAL_CONFIG, indent=2)

    ##############################################
    # connect to database and get the top priority configurations
    ##############################################
    print "connecting to database"
    db_name = 'sqlite:///' + GLOBAL_CONFIG['experiment_db']
    engine = create_engine(db_name, echo=False,
                           connect_args={'check_same_thread': False},
                           poolclass=StaticPool)  # make session available to threads
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    SESSION = DBSession()

    ##############################################
    # Setup logger
    ##############################################
    fname = GLOBAL_CONFIG['checkpoint_path'] + '{}_experiment.log.txt'.format(time.strftime('%Y_%m_%d_%H%M'))
    logging.basicConfig(filename=fname, level=logging.INFO)
    logging.info(json.dumps(GLOBAL_CONFIG, indent=2))

    # Build constant data
    BATCH_DATA = get_batch_data(GLOBAL_CONFIG, subset_num_items=20)  # TODO: change to -1
    EVAL_DATA_TRAIN, EVAL_DATA_VAL = get_eval_data(GLOBAL_CONFIG, subset_train=50, subset_val=25)  # TODO: change to -1
    NUM_ITEMS_TRAIN = 20  # TODO: change to actual number

    main(args)

