import pickle
from net.multimodal.multimodal_solver import MultiModalSolver
from net.multimodal.data_provider.eval_data_class import EvaluationData, EvaluationDataMWQ
from net.multimodal import multimodal_utils as mm_utils
from net.multimodal import experiment
from net.multimodal.evaluation import metrics
import json
import numpy as np


class Report(object):
    """

    """

    def __init__(self, fname):

        self.fname = fname
        self._load_report()
        self._set_exp_config()
        self._set_eval_data()
        self._set_model()
        self._set_solver()

    def _load_report(self):
        with open(self.fname, "rb") as f:
            self.report = pickle.load(f)

    def _set_exp_config(self):
        self.exp_config = self.report['exp_config']
        # things not in report needed for the solver
        self.exp_config['eval_k'] = 10
        self.exp_config['associat_margin'] = 1.  # not in report
        self.exp_config['ck_perform_every'] = 20

    def _set_eval_data(self):
        imgid2region_indices_test = mm_utils.mk_toy_img_id2region_indices(json_fname=self.exp_config['json_path_test'],
                                                                          cnn_fname=self.exp_config[
                                                                              'cnn_full_img_path_test'],
                                                                          num_regions_per_img=1,
                                                                          subset_num_items=-1)

        self.eval_data_test = EvaluationData(json_fname=self.exp_config['json_path_test'],
                                             cnn_fname=self.exp_config['cnn_full_img_path_test'],
                                             img_id2cnn_region_indeces=imgid2region_indices_test,
                                             w2v_vocab_fname=self.exp_config['word2vec_vocab'],
                                             w2v_vectors_fname=self.exp_config['word2vec_vectors'],
                                             external_vocab_fname=self.exp_config['external_vocab'],
                                             subset_num_items=-1)

        # self.eval_data_test_mwq = EvaluationDataMWQ(json_fname=self.exp_config['json_path_test'],
        #                                             cnn_fname=self.exp_config['cnn_full_img_path_test'],
        #                                             img_id2cnn_region_indeces=imgid2region_indices_test,
        #                                             w2v_vocab_fname=self.exp_config['word2vec_vocab'],
        #                                             w2v_vectors_fname=self.exp_config['word2vec_vectors'],
        #                                             external_vocab_fname=self.exp_config['external_vocab'],
        #                                             mwq_aggregator=self.exp_config['mwq_aggregator'],
        #                                             subset_num_items=-1)

    def _set_model(self):
        self.mm_net = experiment.set_model(self.exp_config)
        # Use weigths from the report instead of random init weights
        self.mm_net.params['Wi2s'] = self.report['model']['Wi2s']
        self.mm_net.params['bi2s'] = self.report['model']['bi2s']
        self.mm_net.params['Wsem'] = self.report['model']['Wsem']
        self.mm_net.params['bsem'] = self.report['model']['bsem']

    def _set_solver(self):
        self.solver = MultiModalSolver(model=self.mm_net,
                                       batch_data=None,
                                       eval_data_train=None,
                                       eval_data_val=None,
                                       eval_data_test=self.eval_data_test,
                                       eval_data_test_mwq=None, num_items_train=48689,
                                       exp_config=self.exp_config)

    def get_performance(self, task, metric, K):
        if task == 'i2t':
            performance = self.solver.ck_perform_ranking_img2txt_all_ks(self.eval_data_test, Ks=[K])

        elif task == 't2i':
            performance = self.solver.ck_perform_ranking_img2txt_all_ks(self.eval_data_test, Ks=[K])
        else:
            raise ValueError

        return performance[metric][K]

    def predict_words(self, k=20):
        # top k predicted words
        sim_img_word = self.solver.model.loss(self.eval_data_test, eval_mode=True)
        word_ids_pred = np.argsort(-sim_img_word, axis=1).tolist()

        words_pred = []
        for wids in word_ids_pred:
            words_pred_item = [self.eval_data_test.id2word_ext_vocab[i] for i in wids[0:k]]
            words_pred.append(words_pred_item)

        true_words = self.eval_data_test.true_words_list
        assert len(true_words) == len(words_pred)

        return true_words, words_pred

if __name__ == '__main__':

    import os
    ckpoint_path = '/Users/susanaparis/Documents/Belgium/Chapter4/data/fashion53k/promising_reports/paris/'

    fnames = []
    fnames.append(ckpoint_path + 'report_valf1_0.1166_id_41_hd_500_l_1.0_g_0.0_a_0.0_e_0_p_0.2041_r_0.3163.pkl')
    fnames.append(ckpoint_path + 'report_valf1_0.1214_id_91_hd_1000_l_0.5_g_0.0_a_0.5_e_0_p_0.2109_r_0.3214.pkl')
    fnames.append(ckpoint_path + 'report_valf1_0.1323_id_95_hd_500_l_0.0_g_0.0_a_1.0_e_0_p_0.2358_r_0.3553.pkl')

    # ckpoint_path = '/Users/susanaparis/Documents/Belgium/Chapter4/data/fashion53k/promising_reports/best_all/'
    #
    # fnames.append(ckpoint_path + 'report_valf1_9.07_id_60_hd_1200_l_0.5_g_0.0_a_0.5_e_2_p_15.75_r_23.49_p_5.66_r_2.06_r_1.50_it_260_p_14.98_r_23.61_p_5.57_r_1.39.pkl')
    # fnames.append(ckpoint_path + 'report_valf1_9.04_id_63_hd_500_l_0.0_g_0.0_a_1.0_e_1_p_14.90_r_24.00_p_3.60_r_0.53_r_0.90_it_60_p_14.97_r_25.61_p_4.48_r_0.22.pkl')
    # fnames.append(ckpoint_path + 'report_valf1_8.86_id_76_hd_800_l_1.0_g_0.0_a_0.0_e_1_p_14.67_r_23.31_p_4.17_r_1.77_r_1.50_it_80_p_14.44_r_24.57_p_4.48_r_0.64.pkl')


    # models = ['l1', 'l5', 'a1']
    print "n files: ", len(fnames)
    models = ['a1', 'a5', 'a0']
    assert len(fnames) == len(models)
    i = 0
    for fname in fnames:
        r = Report(fname=fname)
        exp = 'i2t'
        model = models[i]
        Ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 30, 40, 50, 100, 200, 500, 1000]
        print "experiment\t model\t K \t precision\t recall\t f1\t "

        for K in Ks:
            P = r.get_performance(task=exp, metric='P', K=K)
            R = r.get_performance(task=exp, metric='R', K=K)
            F = 2 * P * R / (P + R)
            print "{}\t{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}".format(exp, model, K, P, R, F)

        i += 1

    # print "Precision"
    # print "{:.6f}".format(r.get_performance(task='i2t', metric='P', K=1))
    # print "{:.4f}".format(r.get_performance(task='i2t', metric='P', K=5))
    # print "{:.4f}".format(r.get_performance(task='i2t', metric='P', K=10))
    # print "{:.4f}".format(r.get_performance(task='i2t', metric='P', K=20))
    # print "{:.4f}".format(r.get_performance(task='i2t', metric='P', K=1))
    # print "{:.4f}".format(r.get_performance(task='i2t', metric='P', K=5))
    # print "{:.4f}".format(r.get_performance(task='i2t', metric='P', K=10))
    # print "{:.4f}".format(r.get_performance(task='i2t', metric='P', K=20))
    #
    # print "Recall"
    # print "{:.4f}".format(r.get_performance(task='i2t', metric='R', K=1))
    # print "{:.4f}".format(r.get_performance(task='i2t', metric='R', K=5))
    # print "{:.4f}".format(r.get_performance(task='i2t', metric='R', K=10))
    # print "{:.4f}".format(r.get_performance(task='i2t', metric='R', K=20))
