import numpy as np
import logging
from net import optim
from net.multimodal.evaluation import metrics
import pickle
import time
from net.multimodal import multimodal_utils


class MultiModalSolver(object):
    """

    """

    def __init__(self, model, batch_data, eval_data_train, eval_data_val,
                 num_items_train, exp_config, verbose=True):
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

        """
        self.model = model
        self.batch_data = batch_data
        self.num_items_train = num_items_train  # eval_data_train.X_img.shape[0]  # number of images in training set

        self.eval_data_train = eval_data_train
        self.eval_data_val = eval_data_val
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
        self.batch_size = exp_config['batch_size']
        self.num_epochs = exp_config['num_epochs']

        self.print_every = exp_config['print_every']
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


        # img2txt
        self.train_f1_img2txt_history = []
        self.train_precision_img2txt_history = []
        self.train_recall_img2txt_history = []

        self.val_f1_img2txt_history = []
        self.val_precision_img2txt_history = []
        self.val_recall_img2txt_history = []

        # txt2img
        self.train_f1_txt2img_history = []
        self.train_precision_txt2img_history = []
        self.train_recall_txt2img_history = []

        self.val_f1_txt2img_history = []
        self.val_precision_txt2img_history = []
        self.val_recall_txt2img_history = []

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

        self.batch_data.mk_minibatch(batch_size=self.batch_size, verbose=True)  # TODO: chance verbose to False if you dont want to see the img_ids from the minibatch

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

    def check_performance_img2txt(self, eval_data):
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
        sim_region_word = self.model.loss(eval_data, eval_mode=True)
        y_pred = np.ones(sim_region_word.shape)
        y_pred[sim_region_word < 0] = -1  # when the sim scores are < 0, classification is negative

        p, r, f1 = metrics.precision_recall_f1(y_pred, eval_data.y, raw_scores=False)

        return p, r, f1

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

    def train(self):
        """
        Run optimization to train the model.
        """
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

            self._step()

            if t == 0: loss0 = self.loss_history[0]
            if self.loss_history[t] > 10 * loss0:
                abort_msg = "id_{} \t loss is exploiding. ABORT!".format(self.id)
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
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Create report, Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t + 1 == num_iterations)
            if first_it or last_it or epoch_end:

                # Create report
                report = {}
                report['id'] = self.id
                report['loss_history'] = self.loss_history
                report['iter'] = t
                report['epoch'] = self.epoch
                report['exp_config'] = self.exp_config

                # Evaluate precision, recall and f1 on both tasks
                p_i2t_t, r_i2t_t, f1_i2t_t = self.check_performance_img2txt(self.eval_data_train)  # eval on train

                p_i2t_v, r_i2t_v, f1_i2t_v = self.check_performance_img2txt(self.eval_data_val)  # eval on val

                # p_t2i_t, r_t2i_t, f1_t2i_t = self.check_performance_txt2img(self.X_img_train, self.X_txt_train,
                #                                                             self.y_true_img2txt_train.T)
                #
                # p_t2i_v, r_t2i_v, f1_t2i_v = self.check_performance_txt2img(self.X_img_val, self.X_txt_val,
                #                                                             self.y_true_img2txt_val.T)
                # TODO: look into whether it should be that txt2img and img2txt yeild same results
                # TODO: is check_performance compatible with associat_loss?

                # Add to report
                report['img2txt'] = {}
                report['img2txt']['train_current_performance'] = {'p': p_i2t_t, 'r': r_i2t_t, 'f1': f1_i2t_t}
                report['img2txt']['val_current_performance'] = {'p': p_i2t_v, 'r': r_i2t_v, 'f1': f1_i2t_v}

                # report['txt2img'] = {}
                # report['txt2img']['train_current_performance'] = {'p': p_t2i_t, 'r': r_t2i_t, 'f1': f1_t2i_t}
                # report['txt2img']['val_current_performance'] = {'p': p_t2i_v, 'r': r_t2i_v, 'f1': f1_t2i_v}

                # overall average f1 of both tasks both tasks img2txt & txt2img
                f1_train_score = f1_i2t_t
                f1_val_score = f1_i2t_v

                # img2txt
                self.train_f1_img2txt_history.append(f1_i2t_t)
                self.train_precision_img2txt_history.append(p_i2t_t)
                self.train_recall_img2txt_history.append(r_i2t_t)

                self.val_f1_img2txt_history.append(f1_i2t_v)
                self.val_precision_img2txt_history.append(p_i2t_v)
                self.val_recall_img2txt_history.append(r_i2t_v)

                # txt2img
                # self.train_f1_txt2img_history.append(f1_t2i_t)
                # self.train_precision_txt2img_history.append(p_t2i_t)
                # self.train_recall_txt2img_history.append(r_t2i_t)
                #
                # self.val_f1_txt2img_history.append(f1_t2i_v)
                # self.val_precision_txt2img_history.append(p_t2i_v)
                # self.val_recall_txt2img_history.append(r_t2i_v)

                report['img2txt']['train_history'] = {'p': self.train_precision_img2txt_history,
                                                      'r': self.train_recall_img2txt_history,
                                                      'f1': self.train_f1_img2txt_history}

                report['img2txt']['val_history'] = {'p': self.val_precision_img2txt_history,
                                                    'r': self.val_recall_img2txt_history,
                                                    'f1': self.val_f1_img2txt_history}

                # report['txt2img']['train_history'] = {'p': self.train_precision_txt2img_history,
                #                                       'r': self.train_recall_txt2img_history,
                #                                       'f1': self.train_f1_txt2img_history}
                #
                # report['txt2img']['val_history'] = {'p': self.val_precision_txt2img_history,
                #                                     'r': self.val_recall_txt2img_history,
                #                                     'f1': self.val_f1_txt2img_history}

                # Check if performance is best so far, and if so save report and
                # keep best weights in self.best_params
                if f1_val_score > self.best_val_f1_score:
                    self.best_val_f1_score = f1_val_score
                    self.train_f1_of_best_val = f1_train_score
                    self.best_epoch = self.epoch
                    for k, v in self.model.params.iteritems():
                        self.best_params[k] = v.copy()
                    report['model'] = self.best_params

                    report_fname = 'report_valf1_{0:.4f}_id_{1}_hd_{2}_' \
                                   'l_{3:.1f}_g_{4:.1f}_a_{5:.1f}.pkl'.\
                                   format(self.best_val_f1_score,
                                          self.id,
                                          self.model.h,
                                          self.exp_config['use_local'],
                                          self.exp_config['use_global'],
                                          self.exp_config['use_associat'],
                                          )
                    multimodal_utils.write_report(report_fname, report, self.exp_config, self.best_val_f1_score)

                # get the best train score
                # if f1_train_score > self.best_train_f1_score:
                #     self.best_train_f1_score = f1_train_score

                if last_it:  # save an endreport on last iteration
                    report['model'] = {}  # no need to save weights on end report
                    time_stamp = time.strftime('%Y_%m_%d_%H%M')
                    end_report_fname = self.exp_config['checkpoint_path']
                    end_report_fname += '/id_{0}_end_report_{1}_tr_{2:.4f}_val_{3:.4f}.pkl'.format(self.exp_config['id'],
                                                                                                   time_stamp,
                                                                                                   self.train_f1_of_best_val,
                                                                                                   self.best_val_f1_score,
                                                                                                   )
                    with open(end_report_fname, "wb") as f:
                        pickle.dump(report, f)
                    logging.info("saved report to {}".format(end_report_fname))

                msg = "id_{} ".format(self.exp_config['id'])

                msg += "iter {} \t train_f1 {:.4f} \t val_f1 {:.4f}".format(t+1, f1_train_score, f1_val_score)

                msg += " \t i2t train p {:.4f} r {:.4f} f1 {:.4f}".format(p_i2t_t, r_i2t_t, f1_i2t_t)

                msg += " \t i2t val p {:.4f} r {:.4f} f1 {:.4f}".format(p_i2t_v, r_i2t_v, f1_i2t_v)

                # msg += " \t t2i train p {:.4f} r {:.4f} f1 {:.4f}".format(p_t2i_t, r_t2i_t, f1_t2i_t)
                #
                # msg += " \t t2i val p {:.4f} r {:.4f} f1 {:.4f}".format(p_t2i_v, r_t2i_v, f1_t2i_v)

                logging.info(msg)

                if self.verbose:

                    print msg

                    # print '(Epoch %d / %d) train f1: %f; val_f1: %f' % (
                    #     self.epoch, self.num_epochs, f1_train_score, f1_val_score)
                    #
                    # print "\nImg2Txt Performance"
                    # print "TRAIN f1: {}; \t p: {}; \t r: {}".format(f1_i2t_t, p_i2t_t, r_i2t_t)
                    # print "VAL f1: {}; \t p: {}; \t r: {}".format(f1_i2t_v, p_i2t_v, r_i2t_v)
                    #
                    # print "\nTxt2Img Performance"
                    # print "TRAIN f1: {}; \t p: {}; \t r: {}".format(f1_t2i_t, p_t2i_t, r_t2i_t)
                    # print "VAL f1: {}; \t p: {}; \t r: {}".format(f1_t2i_v, p_t2i_v, r_t2i_v)

                    print "\n"
            self.status = 'done'

        logging.info("id_{} Finished {}\n".format(self.exp_config['id'], time.strftime('%Y_%m_%d_%H%M')))
        # At the end of training swap the best params into the model

        self.model.params = self.best_params



