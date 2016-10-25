
from net.multimodal.evaluation.report import Report
from net.unimodal import unimodal_net

class ReportUnimodal(Report):

    def __init__(self, fname):
        Report.__init__(self, fname)


    def _set_model(self):
        self.mm_net = unimodal_net.set_model(self.exp_config)
        self.mm_net.params['W1'] = self.report['model']['W1']
        self.mm_net.params['b1'] = self.report['model']['b1']
        self.mm_net.params['W2'] = self.report['model']['W2']
        self.mm_net.params['b2'] = self.report['model']['b2']

    def get_performance(self, task, metric, K):
        performance = self.solver.ck_perform_ranking_img2txt_all_ks(self.eval_data_test, Ks=[K])

        return performance[metric][K]


if __name__ == '__main__':
    ckpoint_path = '/Users/susanaparis/Documents/Belgium/Chapter4/data/fashion53k/promising_reports/paris/'

    fnames = []
    fnames.append(ckpoint_path + 'report_valf1_6.38_id_3_hd_800_l_0.0_g_0.0_a_0.0_e_1_p_10.27_r_17.37_p_2.91_r_1.43_r_0.00_it_20_p_10.31_r_18.58_p_2.29_r_0.71.pkl')

    models = ['unimodal']

    i = 0
    for fname in fnames:
        r = ReportUnimodal(fname=fname)
        exp = 'i2t'
        model = models[i]
        Ks = [1, 2, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 100, 200, 500, 1000]
        print "experiment\t model\t K \t precision\t recall\t f1\t "

        for K in Ks:
            P = r.get_performance(task=exp, metric='P', K=K)
            R = r.get_performance(task=exp, metric='R', K=K)
            F = 2 * P * R / (P + R)
            print "{}\t{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}".format(exp, model, K, P, R, F)

        i += 1
