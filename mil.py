import tensorflow as tf
mil_config = {
    'K-shots' : None,
    'maml_tasks_per_batch' : None,
    'split_channels' : None,
    'learning_rate' : None
}

mil_constants = {
    'frames_in_gif' : 50,
    'conv_layers' : 3,
    'action_dim' : 2
}

mil_variables = {
    'stateA' : None,
    'stateB' : None,
    'imgA' : None,
    'imgB' : None,
    'actionA' : None,
    'actionB' : None,
    'not-built' : True,
    'weights' : None
}
def init_network_config(config):
    mil_config['K-shots'] = config['K-shots']
    mil_config['maml_tasks_per_batch'] = config['maml_tasks_per_batch']
    mil_config['split_channels'] = config['split_channels']
    mil_config['learning_rate'] = config['learning_rate']

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
    conv_layers = 3
    fc_layers = 3

    #for i in range(conv_layers):
        
