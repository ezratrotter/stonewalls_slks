#%%

import tensorflow as tf
import os
from tensorflow.python.client import device_lib
#%%

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)