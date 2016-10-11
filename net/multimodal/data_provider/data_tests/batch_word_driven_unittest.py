from net.multimodal.data_provider.batch_class import get_batch_data
from net.multimodal.data_provider.data_tests.test_data_config import exp_config

batch_data = get_batch_data(exp_config)

batch_data.mk_minibatch()
print batch_data.X_img.shape
print batch_data.X_txt.shape
print batch_data.y.shape
print batch_data.y_associat.shape

# TODO: Write assert tests