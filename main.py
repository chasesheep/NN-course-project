from load_data import data_preload
from mil import *

import tensorflow as tf

config = {
    'learning_rate' : 0.001,
    'K-shots': 1,
    'train_samples' : 100,
    'validate_samples' : 20,
    'split_channels' : 30,
    'kernel_size' : 3,
    'fc_layer_size': 200,
    'two_head': False,
    'iterations' : 300,
    'val_interval' : 10,
    'maml_tasks_per_batch': 5,
    'decay': 0.9,
    'strides': [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]],
    'fc_bt': True,
    'clip_min' : -10.0,
    'clip_max' : 10.0
}

constants = {
    'frames_in_gif' : 50,
    'conv_layers' : 3,
    'action_dim' : 2
    #'gif_width': 80,
    #'gif_height': 64
}

graph = tf.Graph()
#GPU options: this part directly migrated from original settings
print('Setting GPU options...')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)
tf_config = tf.ConfigProto(gpu_options = gpu_options)
sess = tf.Session(graph = graph, config = tf_config)

#load data
data_preload(config)
init_network_config(config)
init_network(graph, True)
init_network(graph, False)
