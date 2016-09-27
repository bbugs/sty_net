from net.multimodal import multimodal_utils
import numpy as np

print multimodal_utils.init_random_weights(3,4)
print multimodal_utils.init_random_weights(4).shape

# it returns diferent arrays every time it's called
# print np.random.randn(3,4)
