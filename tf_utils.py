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


def conv2d(img, w, b, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(img, w, strides=strides, padding='SAME') + b


# layer norm and activate
def norm_activate(layer, norm_type=None, activation_fn=tf.nn.relu, id=0, prefix='conv_'):
    if not norm_type: # activate only
        return activation_fn(layer)
    with tf.variable_scope('norm_layer_%s%d' % (prefix, id)) as vs:
        if norm_type == 'layer_norm':  # layer_norm
            try:
                return activation_fn(tf.contrib.layers.layer_norm(layer, center=True, scale=False, scope=vs))
            except ValueError:
                return activation_fn(tf.contrib.layers.layer_norm(layer, center=True, scale=False, scope=vs, reuse=True))
        else:
            raise NotImplementedError('Other types of norm not implemented.')
