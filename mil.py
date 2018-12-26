import tensorflow as tf
import numpy as np
from tf_utils import *

mil_config = {
    # 'K-shots': None,
    # 'maml_tasks_per_batch': None,
    # 'split_channels': None,
    # 'learning_rate': None,
    # 'kernel_size': None,
    # 'strides': None,
    # 'fc_bt': None,
    # 'fc_layer_size': None,
    # 'two_head': None,
    # 'decay': None
}

mil_constants = {
    'frames_in_gif': 50,
    'conv_layers': 3,
    'fc_layers': 3,
    'action_dim': 2,
    'state_dim': 10,
    'bt_dim': 10,
    'gif_width': 80,
    'gif_height': 64,
    # 'conv_out_size': None # will be calculated in init_weights
    # 'conv_out_size_final": None # will be calculated in init_weights, conv_out_size + state_dim [+ bt_dim]
}

mil_variables = {
    'stateA': None,
    'stateB': None,
    'imgA': None,
    'imgB': None,
    'actionA': None,
    'actionB': None,
    'not-built': True,
    'weights': None
}


def init_network_config(config):
    mil_config['K-shots'] = config['K-shots']
    mil_config['maml_tasks_per_batch'] = config['maml_tasks_per_batch']
    mil_config['split_channels'] = config['split_channels']
    mil_config['learning_rate'] = config['learning_rate']
    mil_config['kernel_size'] = config['kernel_size']
    mil_config['strides'] = config['strides']
    mil_config['fc_bt'] = config['fc_bt']
    mil_config['fc_layer_size'] = config['fc_layer_size']
    mil_config['two_head'] = config['two_head']
    mil_config['decay'] = config['decay']


def init_network(graph, training):
    if (mil_variables['not-built']):
        mil_variables['stateA'] = tf.placeholder(tf.float32, name='stateA')
        mil_variables['stateB'] = tf.placeholder(tf.float32, name='stateB')
        mil_variables['imgA'] = tf.placeholder(tf.float32, name='imgA')
        mil_variables['imgB'] = tf.placeholder(tf.float32, name='imgB')
        mil_variables['actionA'] = tf.placeholder(tf.float32, name='actionA')
        mil_variables['actionB'] = tf.placeholder(tf.float32, name='actionB')
        mil_variables['not-built'] = False    
    

def construct_network(training = True):
    stateA = mil_variables['stateA']
    stateB = mil_variables['stateB']
    imgA = mil_variables['imgA']
    imgB = mil_variables['imgB']
    actionA = mil_variables['actionA']
    actionB = mil_variables['actionB']
    
    reuse = not training
    with tf.variable_scope('model', reuse=reuse) as training_scope:
        if training:
            mil_variables['weights'] = init_weights()
        weights = mil_variables['weights']
        lr = mil_config['learning_rate']

        def maml(inputs):
            stateA, stateB, imgA, imgB, actionA, actionB = inputs
            pass

        result = tf.map_fn(maml, elems=(stateA, stateB, imgA, imgB, actionA, actionB))
        #Use map_fn for parallel computation
        #Otherwise the computation is serial

        return result
    

def init_weights():
    weights = {}

    gif_height = mil_constants['gif_height']
    gif_width = mil_constants['gif_width']
    gif_channels = 3
    frames_in_gif = mil_constants['frames_in_gif']

    # conv config
    strides = mil_config['strides']
    kernel_size = mil_config['kernel_size']
    conv_layers = mil_constants['conv_layers']
    if type(kernel_size) is not list:
        kernel_size = [kernel_size] * conv_layers
    split_channels = mil_config['split_channels']
    if type(split_channels) is not list:
        split_channels = [split_channels] * conv_layers

    # downsample factor
    downsample_factor = 1
    for stride in strides:
        downsample_factor *= stride[1]
    mil_constants['conv_out_size'] = int(np.ceil(gif_width / downsample_factor)) \
                                     * int(np.ceil(gif_height / downsample_factor)) \
                                     * split_channels[-1]

    # conv weights
    fan_in = gif_channels
    for i in range(conv_layers):
        weights['wc%d' % (i + 1)] = init_conv_weights_xavier([kernel_size[i], kernel_size[i], fan_in,
                                                              split_channels[i]], name='wc%d' % (i + 1))
        weights['bc%d' % (i + 1)] = init_bias(split_channels[i], name='bc%d' % (i + 1))
        fan_in = split_channels[i]

    # shape of fc in
    in_shape = mil_constants['conv_out_size'] + mil_constants['state_dim']

    # fc bias transformation
    if mil_config['fc_bt']:
        in_shape += mil_constants['bt_dim']
        weights['context'] = tf.get_variable('context', initializer=tf.zeros([mil_constants['bt_dim']], dtype=tf.float32))

    mil_constants['conv_out_size_final'] = in_shape

    # config of fc weights
    fc_layers = mil_constants['fc_layers']
    fc_layer_size = mil_config['fc_layer_size']
    if type(fc_layer_size) is not list:
        fc_layer_size = [fc_layer_size] * (fc_layers - 1)
    fc_layer_size.append(mil_constants['action_dim'])

    # fc weights
    for i in range(fc_layers):
        weights['w_%d' % i] = init_weight([in_shape, fc_layer_size[i]], name='w_%d' % i)
        weights['b_%d' % i] = init_bias([fc_layer_size[i]], name='b_%d' % i)

        # two head
        if i == fc_layers - 1 and mil_config['two_head']:
            weights['w_%d_two_heads' % i] = init_weight([in_shape, fc_layer_size[i]], name='w_%d_two_heads' % i)
            weights['b_%d_two_heads' % i] = init_bias([fc_layer_size[i]], name='b_%d_two_heads' % i)

        in_shape = fc_layer_size[i]

    return weights


def forward(img_input, state_input, weights, meta_testing=False, is_training=True):

    # basic config
    gif_height = mil_constants['gif_height']
    gif_width = mil_constants['gif_width']
    gif_channels = 3
    decay = mil_config['decay']
    strides = mil_config['strides']
    conv_layers = mil_constants['conv_layers']

    # conv layers
    conv_layer = img_input

    for i in range(conv_layers):
        conv_layer = norm_activate(conv2d(conv_layer,
                                          weights['wc%d' % (i + 1)],
                                          weights['bc%d' % (i + 1)],
                                          strides[i]),
                                   'layer_norm',
                                   id=i)

    fc_layer = tf.reshape(conv_layer, [-1, mil_constants['conv_out_size']])

    # fc bt
    if mil_config['fc_bt']:
        context = tf.gather(tf.zeros_like(tf.reshape(img_input,
                                                     [-1, gif_height * gif_width * gif_channels])),
                            axis=1,
                            indices=list(range(mil_constants['bt_dim'])))
        context += weights['context']
        fc_layer = tf.concat(axis=1, values=[fc_layer, context])

    # fc config
    fc_layers = mil_constants['fc_layers']
    fc_layer = tf.concat(axis=1, values=[fc_layer, state_input])

    # fc layers
    for i in range(fc_layers):
        if i == fc_layers - 1 and mil_config['two_head'] and not meta_testing:
            fc_layer = tf.matmul(fc_layer, weights['w_%d_two_heads' % i]) + weights['b_%d_two_heads' % i]
        else:
            fc_layer = tf.matmul(fc_layer, weights['w_%d' % i]) + weights['b_%d' % i]
        if i != fc_layers - 1:
            fc_layer = tf.nn.relu(fc_layer)

    return fc_layer
