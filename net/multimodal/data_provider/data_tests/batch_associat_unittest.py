"""

Test BatchDataAssociat
"""

from net.multimodal.data_provider.batch_class import get_batch_data
from pattern_mining import associat_classifiers
from net.multimodal.data_provider.data_tests import test_data_config
# TODO: decouple from test_data_config exp_config so that you have full control over what happens in the test


exp_config = test_data_config.exp_config
batch_data = get_batch_data(exp_config)

batch_data.mk_minibatch(batch_size=exp_config['batch_size'],
                        verbose=True,
                        debug=True)

y = batch_data.y
y_associat = batch_data.y_associat

print y
print y_associat

# TODO: complete tests with asserts