import tensorflow as tf
import numpy as np

def init_conv_weights_xavier(shape, name=None):
    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    return tf.get_variable(name, list(shape), initializer=conv_initializer, dtype=tf.float32)

def init_weights(shape, name=None):
    shape = tuple(shape)
    weights = np.random.normal(scale=0.01, size=shape).astype('f')
    return tf.get_variable(name, list(shape), initializer=tf.constant_initializer(weights), dtype=tf.float32)

def init_bias(shape, name=None):
    return tf.get_variable(name, initializer=tf.zeros(shape, dtype=tf.float32))
