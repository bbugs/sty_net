
from net.multimodal.data_provider.cnn_data import CnnData
from net.multimodal.data_provider.word2vec_data import Word2VecData
from net.multimodal import multimodal_net
import math


def set_model(exp_config):
    print "setting the model"
    #TODO: make sure that the model is correctly setup with all the values from exp_config
    cnn_fname = exp_config['cnn_regions_path_test']
    w2v_vocab_fname = exp_config['word2vec_vocab']
    w2v_vectors_fname = exp_config['word2vec_vectors']

    img_input_dim = CnnData(cnn_fname=cnn_fname).get_cnn_dim()
    txt_input_dim = Word2VecData(w2v_vocab_fname, w2v_vectors_fname).get_word2vec_dim()

    # hyperparameters
    reg = exp_config['reg']
    hidden_dim = exp_config['hidden_dim']

    # local loss settings
    use_local = exp_config['use_local']
    local_margin = exp_config['local_margin']
    local_scale = exp_config['local_scale']

    # global loss settings
    use_global = exp_config['use_global']
    global_margin = exp_config['global_margin']
    global_scale = exp_config['global_scale']
    smooth_num = exp_config['smooth_num']
    global_method = exp_config['global_method']
    thrglobalscore = exp_config['thrglobalscore']

    # associat loss settings
    use_associat = exp_config['use_associat']

    # weight scale for weight initialzation
    std_img = math.sqrt(2. / img_input_dim)
    std_txt = math.sqrt(2. / txt_input_dim)
    weight_scale = {'img': std_img, 'txt': std_txt}

    # finetuning configuration
    use_finetune_cnn = exp_config['use_finetune_cnn']
    use_finetune_w2v = exp_config['use_finetune_w2v']

    mm_net = multimodal_net.MultiModalNet(img_input_dim, txt_input_dim, hidden_dim, weight_scale,
                                          use_finetune_cnn=use_finetune_cnn,
                                          use_finetune_w2v=use_finetune_w2v,
                                          reg=reg, use_local=use_local,
                                          use_global=use_global, use_associat=use_associat, seed=None)
    # finetuning starts as false and it can be set to true inside
    # the MultiModalSolver after a number of epochs.

    mm_net.set_global_score_hyperparams(global_margin=global_margin,
                                        global_scale=global_scale,
                                        smooth_num=smooth_num,
                                        global_method=global_method,
                                        thrglobalscore=thrglobalscore)

    mm_net.set_local_hyperparams(local_margin=local_margin,
                                 local_scale=local_scale,
                                 do_mil=False)
    # do_mil starts as False and it can be set to True inside
    # MultModalSolver after an number of epochs

    return mm_net
