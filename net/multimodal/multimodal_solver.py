import numpy as np
import logging
from net import optim
from net.multimodal.evaluation import metrics
import pickle
import time
from net.multimodal import multimodal_utils as mutils


class MultiModalSolver(object):
    """

    """

    def __init__(self, model, batch_data, eval_data_train, eval_data_val, eval_data_test, eval_data_test_mwq,
                 num_items_train, exp_config, unimodal=False, verbose=True):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data with the following:
          'X_train': Array of shape (N_train, d_1, ..., d_k) giving training images
          'X_val': Array of shape (N_val, d_1, ..., d_k) giving validation images
          'y_train': Array of shape (N_train,) giving labels for training images
          'y_val': Array of shape (N_val,) giving labels for validation images

        - uselocal: Boolean; indicate if you want to use local loss
        - useglobal: Boolean; indicate if you want to use global loss

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the learning
          rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient during
          training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        - verbose: Boolean; if set to false then no output will be printed during
          training.

        Args:
            unimodal:

        """
        self.model = model
        self.batch_data = batch_data
        self.num_items_train = num_items_train  # eval_data_train.X_img.shape[0]  # number of images in training set
        self.batch_size = self.batch_data.batch_size
        self.unimodal = unimodal

        self.eval_data_train = eval_data_train
        self.eval_data_val = eval_data_val
        self.eval_data_test = eval_data_test
        self.eval_data_test_mwq = eval_data_test_mwq
        # self.X_img_train = eval_data_train.X_img
        # self.X_txt_train = eval_data_train.X_txt
        # self.y_true_img2txt_train = eval_data_train.y_true_img2txt
        #
        # self.X_img_val = eval_data_val.X_img
        # self.X_txt_val = eval_data_val.X_txt
        # self.y_true_img2txt_val = eval_data_val.y_true_img2txt

        self.exp_config = exp_config

        self.use_finetune_cnn = exp_config['use_finetune_cnn']
        self.use_finetune_w2v = exp_config['use_finetune_w2v']
        self.use_mil = exp_config['use_mil']

        self.start_modulation = exp_config['start_modulation']
        self.id = exp_config['id']
        self.Ks = exp_config['Ks']
        self.eval_k = exp_config['eval_k']

        # Train data
        # self.X_img_train = data['X_img_train']
        # self.X_txt_train = data['X_txt_train']
        #
        # self.region2pair_id_train = data['region2pair_id_train']
        # self.word2pair_id_train = data['word2pair_id_train']
        #
        # # Validation data
        # self.X_img_val = data['X_img_val']
        # self.X_txt_val = data['X_txt_val']
        #
        # self.region2pair_id_val = data['region2pair_id_val']
        # self.word2pair_id_val = data['word2pair_id_val']
        #
        # # Data to evaluate f1 on training and validation set
        # self.

        # get optimization parameters
        self.update_rule = exp_config['update_rule']
        self.optim_config = exp_config['optim_config']
        self.lr_decay = exp_config['lr_decay']
        self.num_epochs = exp_config['num_epochs']

        self.print_every = exp_config['print_every']
        self.ck_perform_every = exp_config['ck_perform_every']
        self.verbose = verbose

        self.status = None
        self.best_epoch = None

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):  # check if module optim has the attribute update_rule
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)  # replace the string name with the actual function

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        print "resetting history"
        self.epoch = 0
        self.best_val_f1_score = 0
        self.train_f1_of_best_val = 0
        self.best_params = {}

        self.loss_history = []
        self.performance_history =[]
        self.best_performance = {}

        # img2txt
        # self.train_f1_img2txt_history = []
        # self.train_precision_img2txt_history = []
        # self.train_recall_img2txt_history = []
        #
        # self.val_f1_img2txt_history = []
        # self.val_precision_img2txt_history = []
        # self.val_recall_img2txt_history = []
        #
        # # txt2img
        # self.train_f1_txt2img_history = []
        # self.train_precision_txt2img_history = []
        # self.train_recall_txt2img_history = []
        #
        # self.val_f1_txt2img_history = []
        # self.val_precision_txt2img_history = []
        # self.val_recall_txt2img_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.iteritems()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data

        self.batch_data.mk_minibatch(verbose=False)  # verbose to False if you dont want to see the img_ids from the minibatch

        # Compute loss and gradient
        loss, grads = self.model.loss(self.batch_data, eval_mode=False)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.iteritems():
            dw = grads[p]
            if dw is None:  # when usefinetune is False, some gradients will be None #todo: check if finetuning is done properly
                continue
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def ck_perform_classif(self, eval_data):
        """

        Inputs:
        - X_img: np array of size (n_regions, cnn_dim)
        - X_txt_all_vocab: np array of size (V, word2vec_dim). This is basically the original
        word embeddings.
        - y_true_all_vocab: np array of size (n_regions, V). The i,j element indicates
        whether or not (+1,-1) the ith image region corresponds to the to the jth word.

        You can set y_true_all_vocab to contain +1 only for zappos words
        (or other external visual vocab)
        and thus measure how well the network can retrieve those visual words. But
        this function does not need to know, it just needs +1 or -1 for all region-word pairs
        """
        performance = {}
        sim_region_word = self.model.loss(eval_data, eval_mode=True)
        y_pred = np.ones(sim_region_word.shape)
        y_pred[sim_region_word < 0] = -1  # when the sim scores are < 0, classification is negative

        p, r, f1 = metrics.precision_recall_f1(y_pred, eval_data.y, raw_scores=False)
        performance['P'] = p
        performance['R'] = r
        performance['F'] = f1
        return performance

    # def check_performance_txt2img(self, eval_data):
    #
    #     """
    #     Inputs:
    #
    #     - X_img_target: np array of size (n_imgs_in_target_collection, cnn_dim)
    #     - X_txt_zappos_only: np array of size (V_zappos, word2vec_dim)
    #     - y_true_all_vocab: np array of size (V_zappos, n_regions_in_target). The i,j element indicates
    #     whether or not (+1,-1) the ith zappos word corresponds to the jth image.
    #
    #     """
    #     sim_word_region = self.model.loss(eval_data, eval_mode=True).T
    #     y_pred = np.ones(sim_word_region.shape)
    #     y_pred[sim_word_region < 0] = -1  # when the sim scores are < 0, classification is negative
    #
    #     p, r, f1 = metrics.precision_recall_f1(y_pred, eval_data.y_true_txt2img, raw_scores=False)
    #
    #     return p, r, f1

    # def check_accuracy(self, X, y, num_samples=None, batch_size=100):
    #     """
    #     Check accuracy of the model on the provided data.
    #
    #     Inputs:
    #     - X: Array of data, of shape (N, d_1, ..., d_k)
    #     - y: Array of labels, of shape (N,)
    #     - num_samples: If not None, subsample the data and only test the model
    #       on num_samples datapoints.
    #     - batch_size: Split X and y into batches of this size to avoid using too
    #       much memory.
    #
    #     Returns:
    #     - acc: Scalar giving the fraction of instances that were correctly
    #       classified by the model.
    #     """
    #
    #     # Maybe subsample the data
    #     N = X.shape[0]
    #
    #     # Compute predictions in batches
    #     num_batches = N / batch_size
    #     if N % batch_size != 0:
    #         num_batches += 1
    #     y_pred = []
    #     for i in xrange(num_batches):
    #         start = i * batch_size
    #         end = (i + 1) * batch_size
    #         scores = self.model.loss(X[start:end])
    #         y_pred.append(np.argmax(scores, axis=1))
    #     y_pred = np.hstack(y_pred)
    #     acc = np.mean(y_pred == y)
    #
    #     return acc

    def ck_perform_ranking_img2txt_all_ks(self, eval_data, Ks):
        sim_region_word = self.model.loss(eval_data, eval_mode=True)
        word_ids_pred = np.argsort(-sim_region_word, axis=1).tolist()
        true_word_ids = eval_data.true_word_ids_list
        performance = metrics.avg_prec_recall_all_ks(true_word_ids, word_ids_pred, Ks)
        # {'R': {1: 0.27083, 5: 0.54166}, 'P': {1: 0.75, 5: 0.35, 'F': {1: 0.3916, 5: 0.4186}}}
        return performance

    def ck_perform_ranking_txt2img_all_ks(self, eval_data, Ks):
        sim_word_region = self.model.loss(eval_data, eval_mode=True).T
        cnn_indices_pred = np.argsort(-sim_word_region, axis=1).tolist()
        true_cnn_indices = eval_data.true_cnn_indices
        performance = metrics.avg_prec_recall_all_ks(true_cnn_indices, cnn_indices_pred, Ks)
        # {'R': {1: 0.27083, 5: 0.54166}, 'P': {1: 0.75, 5: 0.35, 'F': {1: 0.3916, 5: 0.4186}}}
        return performance

    def compute_performance(self):
        # Evaluate classification performance precision, recall and f1.
        # Same values for both tasks when doing classfication performance.
        classif_performance_train = self.ck_perform_classif(self.eval_data_train)  # eval on train
        classif_performance_val = self.ck_perform_classif(self.eval_data_val)  # eval on val
        classif_performance_test = self.ck_perform_classif(self.eval_data_test)

        rank_performance_i2t_train = self.ck_perform_ranking_img2txt_all_ks(self.eval_data_train, self.Ks)
        rank_performance_i2t_val = self.ck_perform_ranking_img2txt_all_ks(self.eval_data_val, self.Ks)
        rank_performance_i2t_test = self.ck_perform_ranking_img2txt_all_ks(self.eval_data_test, self.Ks)

        rank_performance_t2i_train = self.ck_perform_ranking_txt2img_all_ks(self.eval_data_train, self.Ks)
        rank_performance_t2i_val = self.ck_perform_ranking_txt2img_all_ks(self.eval_data_val, self.Ks)
        rank_performance_t2i_test = self.ck_perform_ranking_txt2img_all_ks(self.eval_data_test, self.Ks)

        if not self.unimodal:
            rank_performance_mwq_t2i_test = self.ck_perform_ranking_txt2img_all_ks(self.eval_data_test_mwq, self.Ks)
        else:
            rank_performance_mwq_t2i_test = None

        performance = {}
        performance['classification'] = {}
        performance['classification']['train'] = classif_performance_train
        performance['classification']['val'] = classif_performance_val
        performance['classification']['test'] = classif_performance_test

        performance['ranking'] = {}
        performance['ranking']['i2t'] = {}
        performance['ranking']['i2t']['train'] = rank_performance_i2t_train
        performance['ranking']['i2t']['val'] = rank_performance_i2t_val
        performance['ranking']['i2t']['test'] = rank_performance_i2t_test

        performance['ranking']['t2i'] = {}
        performance['ranking']['t2i']['train'] = rank_performance_t2i_train
        performance['ranking']['t2i']['val'] = rank_performance_t2i_val
        performance['ranking']['t2i']['test'] = rank_performance_t2i_test

        performance['ranking']['t2i']['mwq_test'] = rank_performance_mwq_t2i_test

        return performance

    @staticmethod
    def get_P_R_F_from_performance(performance, mode, task, split, K):
        """
        Args:
            performance:
            mode: 'ranking' or 'classification'
            task: 'i2t' or 't2i'
            split: 'train', 'val', or 'test'
            K: int

        Returns:
            P, R, F: precision, recall and F1
        """

        if mode == 'classification':
            P = performance[mode][split]['P']*100
            R = performance[mode][split]['R']*100
            F = performance[mode][split]['F']*100
            return P, R, F
        elif mode == 'ranking':
            P = performance[mode][task][split]['P'][K]*100
            R = performance[mode][task][split]['R'][K]*100
            F = performance[mode][task][split]['F'][K]*100
            return P, R, F
        else:
            raise ValueError("mode should be either classification or ranking")

    def mk_status_msg(self, iter_num, performance):

        msg = "\t id_{} \t epoch_{} \t".format(self.exp_config['id'], self.epoch)
        msg += "iter: {} \t".format(iter_num)
        msg += "reg: {} \t".format(self.exp_config['reg'])
        msg += "lr: {} \t".format(self.exp_config['optim_config']['learning_rate'])
        msg += "hd: {} \n".format(self.exp_config['hidden_dim'])

        mode = 'ranking'
        msg += mode + '\n'
        tasks = ['i2t', 't2i']
        splits = ['val', 'train', 'test']
        ks = [1, 5, 10, 20, 100]

        for task in tasks:
            for split in splits:
                for k in ks:
                    P, R, F = self.get_P_R_F_from_performance(performance, mode, task, split, k)
                    msg += "{0} {1} \t P{2} {3:.1f} \t R{2} {4:.1f} \t".\
                        format(task, split, k, P, R, )
                msg += '\n'
            msg += '\n'
        # msg += '\n'

        if not self.unimodal:
            task = 't2i'
            split = 'mwq_test'
            for k in ks:
                P, R, F = self.get_P_R_F_from_performance(performance, mode, task=task, split=split, K=k)
                msg += "{0} {1} \t P{2} {3:.1f} \t R{2} {4:.1f} \t".\
                    format(task, split, k, P, R, )
            msg += '\n\n'

        mode = 'classification'
        msg += 'classif' + '\n'
        task = None  # same for both tasks
        k = None  # k is not relevant in classfication mode
        for split in splits:
            P, R, F = self.get_P_R_F_from_performance(performance, mode, task, split, k)
            msg += "i2t {0} \t P {1:.1f} \t R {2:.1f}".format(split, P, R)
            msg += "\n"
        return msg

    def train(self):
        """
        Run optimization to train the model.
        """
        self._reset()
        logging.info("id_{} {} started training".format(self.id, time.strftime('%Y_%m_%d_%H%M')))
        iterations_per_epoch = max(self.num_items_train / self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        has_modulation_started = False
        for t in xrange(num_iterations):

            # Modulate do_mil and finetuning
            if not has_modulation_started and self.epoch > self.start_modulation * self.num_epochs:
                # TODO: print to screen and add to logger whether these have been started
                if self.use_mil:
                    self.model.do_mil = True
                    logging.info("id_{} do_mil has started".format(self.id))
                if self.use_finetune_cnn:
                    self.model.finetune_cnn = True
                    logging.info("id_{} finetune_cnn has started".format(self.id))
                if self.use_finetune_w2v:
                    self.model.finetune_w2v = True
                    logging.info("id_{} finetune_w2v has started".format(self.id))
                has_modulation_started = True

            self._step()  # The magic happens here

            if t == 0: loss0 = self.loss_history[0]
            if self.loss_history[t] > 1e10 * loss0:
                abort_msg = "id_{} \t epoch_{} \t reg_{} \t lr_{} \t hd_{} \t loss is exploiding. ABORT!".\
                    format(self.id, self.epoch, self.exp_config['reg'],
                           self.exp_config['optim_config']['learning_rate'],
                           self.exp_config['hidden_dim'])
                print abort_msg
                logging.info(abort_msg)
                self.status = 'aborted'
                break

            # Maybe print training loss
            if t % self.print_every == 0:
                msg = 'id_{} \t Epoch: {} / {}. \t Iter {} / {} \t loss: {:.6f}'.format(self.id,
                    self.epoch, self.num_epochs, t + 1, num_iterations, self.loss_history[-1])
                logging.info(msg)
                if self.verbose:
                    print msg

            # At the end of every epoch, increment the epoch counter and decay the
            # learning rate.
            epoch_end = t % (iterations_per_epoch - 1) == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Create report, Check train and val accuracy on the first iteration, the last
            # iteration, at the end of each epoch and at ck_perform_every.
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            ck_perf = (t % self.ck_perform_every) == 0
            if first_it or last_it or epoch_end or ck_perf:

                # Create report
                report = {}
                report['id'] = self.id
                report['loss_history'] = self.loss_history
                report['iter'] = t
                report['epoch'] = self.epoch
                report['exp_config'] = self.exp_config

                # compute peformance
                performance = self.compute_performance()

                # Take the avg of f1@K for img2txt and txt2img
                f1_val_score_at_eval_k = 100 * (0.5 * performance['ranking']['i2t']['val']['F'][self.eval_k] +
                                    0.5 * performance['ranking']['t2i']['val']['F'][self.eval_k])

                f1_train_score_at_eval_k = 100 * (0.5 * performance['ranking']['i2t']['train']['F'][self.eval_k] +
                                      0.5 * performance['ranking']['t2i']['train']['F'][self.eval_k])

                i2t_test_prec_at_eval_k = 100 * performance['ranking']['i2t']['test']['P'][self.eval_k]
                i2t_test_recall_at_eval_k = 100 * performance['ranking']['i2t']['test']['R'][self.eval_k]

                t2i_test_prec_at_eval_k = 100 * performance['ranking']['t2i']['test']['P'][self.eval_k]
                t2i_test_recall_at_eval_k = 100 * performance['ranking']['t2i']['test']['R'][self.eval_k]

                if not self.unimodal:
                    t2i_mwq_test_recall_at_eval_k = 100 * performance['ranking']['t2i']['mwq_test']['R'][self.eval_k]
                else:
                    t2i_mwq_test_recall_at_eval_k = 0.0000

                # TODO: is check_performance compatible with associat_loss?

                self.performance_history.append((self.epoch, performance))
                # Add to report
                report['performance'] = performance
                report['performance_history'] = self.performance_history


                # Check if performance is best so far, and if so save report and
                # keep best weights in self.best_params
                if f1_val_score_at_eval_k > self.best_val_f1_score:
                    self.best_val_f1_score = f1_val_score_at_eval_k
                    self.train_f1_of_best_val = f1_train_score_at_eval_k
                    self.best_epoch = self.epoch
                    for k, v in self.model.params.iteritems():
                        self.best_params[k] = v.copy()
                    report['model'] = self.best_params

                    report_fname = 'report_valf1_{0:.2f}_id_{1}_hd_{2}_' \
                                   'l_{3:.1f}_g_{4:.1f}_a_{5:.1f}_' \
                                   'e_{6}_p_{7:.2f}_r_{8:.2f}_' \
                                   'p_{9:.2f}_r_{10:.2f}_r_{11:.2f}_it_{12}_' \
                                   'p_{13:.2f}_r_{14:.2f}_' \
                                   'p_{15:.2f}_r_{16:.2f}.pkl'.\
                                   format(self.best_val_f1_score,
                                          self.id,
                                          self.model.h,
                                          self.exp_config['use_local'],
                                          self.exp_config['use_global'],
                                          self.exp_config['use_associat'],
                                          self.epoch,
                                          i2t_test_prec_at_eval_k,
                                          i2t_test_recall_at_eval_k,
                                          t2i_test_prec_at_eval_k,
                                          t2i_test_recall_at_eval_k,
                                          t2i_mwq_test_recall_at_eval_k,
                                          t,
                                          100 * performance['ranking']['i2t']['val']['P'][self.eval_k],
                                          100 * performance['ranking']['i2t']['val']['R'][self.eval_k],
                                          100 * performance['ranking']['t2i']['val']['P'][self.eval_k],
                                          100 * performance['ranking']['t2i']['val']['R'][self.eval_k])

                    mutils.write_report_for_exp_id(report_fname, report, self.exp_config, self.best_val_f1_score)

                if last_it:  # save an endreport on last iteration
                    report['model'] = {}  # no need to save weights on end report
                    time_stamp = time.strftime('%Y_%m_%d_%H%M')
                    end_report_fname = self.exp_config['checkpoint_path']
                    end_report_fname += '/id_{0}_end_report_{1}_tr_{2:.2f}_val_{3:.2f}.pkl'.format(self.exp_config['id'],
                                                                                                   time_stamp,
                                                                                                   self.train_f1_of_best_val,
                                                                                                   self.best_val_f1_score,
                                                                                                   )
                    with open(end_report_fname, "wb") as f:
                        pickle.dump(report, f)
                    logging.info("saved report to {}".format(end_report_fname))

                msg = self.mk_status_msg(iter_num=t, performance=performance)
                print "avg_i2t_t2i_val_f1_at{0}:\t {1:.2f}".format(self.eval_k, f1_val_score_at_eval_k)
                print msg
                logging.info(msg)

            self.status = 'done'

        logging.info("id_{} Finished {}\n".format(self.exp_config['id'], time.strftime('%Y_%m_%d_%H%M')))
        # At the end of training swap the best params into the model

        self.model.params = self.best_params



