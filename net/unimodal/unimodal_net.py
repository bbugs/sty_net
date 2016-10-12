

from net.layer_utils import *
from net.multimodal import multimodal_utils as mm_utils
import math
from net.multimodal.data_provider.cnn_data import CnnData
from net.multimodal.data_provider.word2vec_data import Word2VecData
from net.multimodal.data_provider.vocab_data import Vocabulary


class UniModalNet(object):
    """

    """

    def __init__(self, input_dim=4096, hidden_dim=100, num_classes=211,
                 weight_scale=1e-3, reg=0.0):

        self.params = {}
        self.reg = reg
        self.h = hidden_dim
        self.num_classes = num_classes

        # Initialize weights
        self.params['W1'] = weight_scale * mm_utils.init_random_weights(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)

        self.params['W2'] = weight_scale * mm_utils.init_random_weights(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, data, eval_mode):
        """
        Args:
            data:  minibatch or eval data
            eval_mode:

        Returns:

        """
        X_img = data.X_img
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        ############################################################
        # Forward Pass
        ############################################################

        out1, cache1 = affine_relu_forward(X_img, W1, b1)
        img_word_scores, scores_cache = affine_forward(out1, W2, b2)

        if eval_mode:
            return img_word_scores

        y = data.y
        data_loss, dscores = svm_two_classes(img_word_scores, y, delta=1., do_mil=False, normalize=True)

        reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) +
                                     np.sum(W2 * W2))
        loss = data_loss + reg_loss  # add the data loss and reg loss

        ############################################################
        # Backward Pass
        ############################################################
        grads = {}

        dx2, dW2, db2 = affine_backward(dscores, scores_cache)
        dx, dW1, db1 = affine_relu_backward(dx2, cache1)

        dW2 += self.reg * W2
        dW1 += self.reg * W1

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads


def set_model(exp_config):
    print "setting the model"
    #TODO: make sure that the model is correctly setup with all the values from exp_config
    cnn_fname = exp_config['cnn_full_img_path_test']  # just to get cnn dim, ok to use test
    w2v_vocab_fname = exp_config['word2vec_vocab']
    w2v_vectors_fname = exp_config['word2vec_vectors']

    img_input_dim = CnnData(cnn_fname=cnn_fname).get_cnn_dim()
    txt_input_dim = Word2VecData(w2v_vocab_fname, w2v_vectors_fname).get_word2vec_dim()

    external_vocab = Vocabulary(exp_config['external_vocab'])  # zappos
    external_vocab_words = external_vocab.get_vocab()  # list of words in vocab
    num_classes = len(external_vocab_words)

    # hyperparameters
    reg = exp_config['reg']
    hidden_dim = exp_config['hidden_dim']

    # weight scale for weight initialzation
    std_img = math.sqrt(2. / img_input_dim)
    std_txt = math.sqrt(2. / txt_input_dim)
    weight_scale = {'img': std_img, 'txt': std_txt}

    # finetuning configuration
    use_finetune_cnn = exp_config['use_finetune_cnn']
    use_finetune_w2v = exp_config['use_finetune_w2v']

    mm_net = UniModalNet(input_dim=img_input_dim,
                         hidden_dim=hidden_dim,
                         num_classes=num_classes,
                         weight_scale=std_img,
                         reg=reg)

    # finetuning starts as false and it can be set to true inside
    # the MultiModalSolver after a number of epochs.

    # mm_net.set_global_score_hyperparams(global_margin=global_margin,
    #                                     global_scale=global_scale,
    #                                     smooth_num=smooth_num,
    #                                     global_method=global_method,
    #                                     thrglobalscore=thrglobalscore)


    # do_mil starts as False and it can be set to True inside
    # MultModalSolver after an number of epochs

    return mm_net
