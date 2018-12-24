
mil_config = {
    'K-shots' : None,
    'maml_tasks_per_batch' : None,
    'split_channels' : None,
}

mil_constants = {
    'frames_in_gif' : 50,
    'conv_layers' : 3,
    'action_dim' : 2
}
def init_network_config(config):
    mil_config['K-shots'] = config['K-shots']
    mil_config['maml_tasks_per_batch'] = config['maml_tasks_per_batch']
    mil_config['split_channels'] = config['split_channels']
    mil_config['learning_rate'] = config['learning_rate']

def init_network(graph, training):
    pass

    
