import tensorflow as tf
import numpy as np
from tf_utils import *
from load_data import read_gif

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
    'gif_channels' : 3
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
    'weights': None
}


def init_network_config(config):
    mil_config['K-shots'] = config['K-shots']
    mil_config['maml_tasks_per_batch'] = config['maml_tasks_per_batch']
    mil_config['split_channels'] = config['split_channels']
    mil_config['learning_rate'] = config['learning_rate']
    mil_config['meta_learning_rate'] = config['meta_learning_rate']
    mil_config['kernel_size'] = config['kernel_size']
    mil_config['strides'] = config['strides']
    mil_config['fc_bt'] = config['fc_bt']
    mil_config['fc_layer_size'] = config['fc_layer_size']
    mil_config['two_head'] = config['two_head']
    mil_config['decay'] = config['decay']
    mil_config['clip_min'] = config['clip_min']
    mil_config['clip_max'] = config['clip_max']


def init_network(graph, training):
    with graph.as_default():
        result = construct_network(training)
        outputAs, outputBs, lossAs, lossBs = result

        batch_size = mil_config['maml_tasks_per_batch']
        lossA = tf.reduce_sum(lossAs) / tf.to_float(batch_size)
        lossB = tf.reduce_sum(lossBs) / tf.to_float(batch_size)

        lr = mil_config['meta_learning_rate']

        if training:
            mil_variables['train_lossA'] = lossA
            mil_variables['train_lossB'] = lossB
            train_op = tf.train.AdamOptimizer(lr).minimize(lossB)
            mil_variables['train_op'] = train_op
        else:
            mil_variables['val_lossA'] = lossA
            mil_variables['val_lossB'] = lossB

def construct_network(training = True):
    
    if (training):
        length = mil_config['maml_tasks_per_batch'] * mil_config['K-shots']
        mil_variables['stateA'] = tf.placeholder(tf.float32, name='stateA')
        mil_variables['stateB'] = tf.placeholder(tf.float32, name='stateB')
        mil_variables['img_namesA'] = tf.placeholder(tf.string, shape=(length), name='img_namesA')
        mil_variables['img_namesB'] = tf.placeholder(tf.string, shape=(length), name='img_namesB')
        mil_variables['actionA'] = tf.placeholder(tf.float32, name='actionA')
        mil_variables['actionB'] = tf.placeholder(tf.float32, name='actionB')
    
    stateA = mil_variables['stateA']
    stateB = mil_variables['stateB']
    #imgA = mil_variables['imgA']
    #imgB = mil_variables['imgB']
    imgA = tf.map_fn(read_gif, mil_variables['img_namesA'], dtype=tf.float32)
    imgB = tf.map_fn(read_gif, mil_variables['img_namesB'], dtype=tf.float32)

    #imgA = tf.reshape(imgA, shape=(-1, mil_constants['gif_height'], mil_constants['gif_width'], mil_constants['gif_channels']))
    #imgB = tf.reshape(imgB, shape=(-1, mil_constants['gif_height'], mil_constants['gif_width'], mil_constants['gif_channels']))
    
    actionA = mil_variables['actionA']
    actionB = mil_variables['actionB']

    '''imgA = imgA.reshape((-1, mil_constants['gif_height'], \
                                 mil_constants['gif_width'], 3))
    imgB = imgB.reshape((-1, mil_constants['gif_height'], \
                                 mil_constants['gif_width'], 3))'''

    print('Constructing network...')
    reuse = not training
    with tf.variable_scope('model', reuse=reuse) as training_scope:
        if training:
            mil_variables['weights'] = init_weights()
        weights = mil_variables['weights']
        lr = mil_config['learning_rate']

        def maml(inputs):
            stateA, stateB, imgA, imgB, actionA, actionB = inputs
            fast_weights = dict(zip(weights.keys(), weights.values()))

            #inner update
            outputA = forward(imgA, stateA, fast_weights, is_training=training)
            lossA = euclidean_loss_layer(outputA, actionA)

            #calculate gradient
            grads = tf.gradients(lossA, list(fast_weights.values()))

            #clip gradient(optional)
            clip_min = mil_config['clip_min']
            clip_max = mil_config['clip_max']
            grads = [tf.clip_by_value(item, clip_min, clip_max) for item in grads]

            #update fast_weights
            gradients = dict(zip(fast_weights.keys(), grads))
            fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - lr * gradients[key] for key in fast_weights.keys()]))

            #outer update
            outputB = forward(imgB, stateB, fast_weights, is_training=training)
            lossB = euclidean_loss_layer(outputB, actionB)

            return [outputA, outputB, lossA, lossB]
        
        out_dtype = [tf.float32, tf.float32, tf.float32, tf.float32]
        result = tf.map_fn(maml, elems=(stateA, stateB, imgA, imgB, \
                                        actionA, actionB), dtype = out_dtype)
        #Use map_fn for parallel computation
        #Otherwise the computation is serial
        return result
    

def init_weights():
    print('initializing weights...')
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
