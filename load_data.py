import numpy as np
import pickle
import os

load_data_config = {
    'reach_path' : 'data/sim_vision_reach',
    'test_path' : 'data',
    'push_path' : 'data',      #to be modified
    'shuffle' : True,
    'state_vec_len' : 10,
    'action_vec_len' : 2,

    #variables below are initialized after program begins running
    
    'total_demos' : 0,       #number of demos in the directory
    'size' : 0,              #total size of training and validation samples
    'train_size': 0,         #training samples
    'val_size' : 0,          #validation_samples
    'iterations' : 0,
    'val_interval' : 0,

    #normalization factors
    'state_scale' : None,
    'state_bias' : None
}

load_data_constants = {
    'gif_dir_prefix' : 'color_',
    'frames_in_gif' : 50
}

load_data_variables = {
    'selected_index' : None,
    'selected_data' : None,
    'train_index': None,
    'train_data': None,
    'val_index': None,
    'val_data': None,
    'train_gif_dirs' : None,
    'val_gif_dirs' : None
}
#selected_data = None
#selected_index = None

def init_load_data_config(config):
    load_data_config['size'] = config['train_samples'] + config['validate_samples']
    load_data_config['train_size'] = config['train_samples']
    load_data_config['val_size'] = config['validate_samples']
    load_data_config['iterations'] = config['iterations']
    load_data_config['val_interval'] = config['val_interval']

def load_states_and_actions():
    
    gif_dir = load_data_config['reach_path']
    size = load_data_config['size']
    print("Randomly selecting " + str(size) + " demos...")
    
    files = [os.path.join(gif_dir, f) for f in os.listdir(gif_dir) if f.endswith('pkl')]
    
    load_data_config['total_demos'] = len(files)
    demo_index = np.array(range(load_data_config['total_demos']))
    np.random.shuffle(demo_index)

    selected_index = demo_index[:size]

    print('Loading pickle files...')
    selected_data = [unpickle(files[i]) for i in selected_index]
    normalize_states(selected_data)
    load_data_variables['selected_index'] = selected_index
    load_data_variables['selected_data'] = selected_data

def split_selected_data():
    selected_index = load_data_variables['selected_index']
    selected_data  = load_data_variables['selected_data']
    train_size = load_data_config['train_size']
    val_size = load_data_config['val_size']
    load_data_variables['train_index'] = selected_index[:train_size]
    load_data_variables['val_index'] = selected_index[-val_size:]
    load_data_variables['train_data'] = selected_data[:train_size]
    load_data_variables['val_data'] = selected_data[-val_size:]

def load_gif_dirs():
    print('Loading gif-directory names')
    gif_dir = load_data_config['reach_path']
    prefix = load_data_constants['gif_dir_prefix']
    train_index = load_data_variables['train_index']
    val_index = load_data_variables['val_index']
    train_gif_dirs = [os.path.join(gif_dir, prefix + str(i)) for i in train_index]
    val_gif_dirs = [os.path.join(gif_dir, prefix + str(i)) for i in val_index]
    load_data_variables['train_gif_dirs'] = train_gif_dirs
    load_data_variables['val_gif_dirs'] = val_gif_dirs
    #print(val_gif_dirs)

def unpickle(name):
    if name.endswith('.pkl'):
        file = open(name, 'rb')
        try:
            content = pickle.load(file, encoding='latin1')
            return content
        except IOError:
            print("IOError while unpickling!")
        return None
    else:
        print('Filename error!')
        return None

def gen_batch_filenames():
    iterations = load_data_config['iterations']
    val_interval = load_data_config['val_interval']
    #for i in range(iterations):
        
        
    
def normalize_states(data):
    #To evaluate training result, the normalizing factor
    #must be calculated only according to the training samples

    #Each pickle files contains several gifs
    #Each gif contains 50 frames
    #Each frame has a corresponding length-10 state vector

    #Here, we normalize the data according to each position of state vectors
    l = load_data_config['state_vec_len']
    
    state_vectors = np.vstack((data[i]['demoX'] for i in range(load_data_config['size'])))
    state_vectors = np.reshape(state_vectors, (-1, l))
    scale = np.maximum(np.std(state_vectors, axis = 0), 0.001)
    scale_matrix = np.diag(1.0 / scale)
    state_vectors = np.matmul(state_vectors, scale_matrix)
    avg = np.mean(state_vectors, axis = 0)

    for i in range(load_data_config['size']):
        data[i]['demoX'] = np.reshape(data[i]['demoX'], (-1, l))
        data[i]['demoX'] = np.matmul(data[i]['demoX'], scale_matrix) - avg
        data[i]['demoX'] = np.reshape(data[i]['demoX'],(-1, load_data_constants['frames_in_gif'], l))

    print('Done state-vector normalization')
    #To be modified...
    pass

def generate_batch():
    pass

def data_preload(config):
    init_load_data_config(config)
    load_states_and_actions()
    split_selected_data()
    load_gif_dirs()
    gen_batch_filenames()
