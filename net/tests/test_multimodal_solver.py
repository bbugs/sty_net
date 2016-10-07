
import logging
import time
from net.multimodal.data_provider.eval_data_class import get_eval_data
from net.multimodal.data_provider.batch_class import get_batch_data
from net.multimodal.multimodal_solver import MultiModalSolver
from net.multimodal.data_provider.data_tests import test_data_config
from net.multimodal import experiment


##############################################
# Setup logger
##############################################
fname = test_data_config.exp_config['checkpoint_path'] + '{}_experiment.log.txt'.format(time.strftime('%Y_%m_%d_%H%M'))
logging.basicConfig(filename=fname, level=logging.INFO)

num_items_train = test_data_config.exp_config['num_items_train']

##############################################
# get batch data
##############################################
batch_data = get_batch_data(exp_config=test_data_config.exp_config)

##############################################
# get eval data
##############################################
eval_data_train, eval_data_val, eval_data_test, eval_data_test_mwq = get_eval_data(exp_config=test_data_config.exp_config)

##############################################
# set neural network architecture
##############################################
mm_net = experiment.set_model(exp_config=test_data_config.exp_config)

##############################################
# Train model with solver
##############################################
solver = MultiModalSolver(mm_net, batch_data,
                          eval_data_train, eval_data_val,
                          eval_data_test, eval_data_test_mwq,
                          num_items_train, test_data_config.exp_config, verbose=True)

solver.train()


## Later
# for i in xrange(len(lr)):
#     for j in xrange(len(reg)):
#         model = TwoLayerNet(hidden_dim=300, reg=reg[j], weight_scale=1e-2)
#         solver = Solver(model, data,update_rule='rmsprop', optim_config={'learning_rate': lr[i],},
#                       lr_decay=0.95,
#                       num_epochs=10, batch_size=1024,
#                       print_every=1024)
#         solver.train()
#         if solver.best_val_acc > best_val:
#             best_model = model
#             best_val = solver.best_val_acc
#         results[lr[i],reg[j]] = solver.best_val_acc