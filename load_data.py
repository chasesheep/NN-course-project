import numpy as np
import pickle
import os
import tensorflow as tf
from natsort import natsorted

load_data_config = {
    'reach_path' : 'data\\sim_vision_reach',
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
    'tasks_per_batch' : 0,
    'K-shots' : 0,

    #normalization factors
    'state_scale' : None,
    'state_bias' : None
}

load_data_constants = {
    'gif_dir_prefix' : 'color_',
    'frames_in_gif' : 50,
    'img_height' : 64,
    'img_width' : 80,
    'color_channels' : 3
}

load_data_variables = {
    'selected_index' : None,
    'selected_data' : None,
    'train_index': None,
    'train_data': None,
    'val_index': None,
    'val_data': None,
    'train_gif_dirs' : None,
    'val_gif_dirs' : None,
    'all_train_filenames' : None,
    'all_val_filenames' : None,
    'train_itr' : None,
    'val_itr' : None
}
#selected_data = None
#selected_index = None

def init_load_data_config(config):
    load_data_config['size'] = config['train_samples'] + config['validate_samples']
    load_data_config['train_size'] = config['train_samples']
    load_data_config['val_size'] = config['validate_samples']
    load_data_config['iterations'] = config['iterations']
    load_data_config['val_interval'] = config['val_interval']
    load_data_config['tasks_per_batch'] = config['maml_tasks_per_batch']
    load_data_config['K-shots'] = config['K-shots']

def load_states_and_actions():
    
    gif_dir = load_data_config['reach_path']
    size = load_data_config['size']
    print("Randomly selecting " + str(size) + " demos...")
    
    files = natsorted([os.path.join(gif_dir, f) for f in os.listdir(gif_dir) if f.endswith('pkl')])
    
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
    print('Generating batch filenames...')
    iterations = load_data_config['iterations']
    val_interval = load_data_config['val_interval']
    batch_size = load_data_config['tasks_per_batch']
    train_size = load_data_config['train_size']
    val_size = load_data_config['val_size']
    shots = load_data_config['K-shots'] * 2
    train_gif_dirs = load_data_variables['train_gif_dirs']
    val_gif_dirs = load_data_variables['val_gif_dirs']

    all_train_filenames = []
    all_val_filenames = []

    all_train_index = []
    all_val_index = []

    all_train_gif_index = []
    all_val_gif_index = []
    
    for i in range(iterations):
        rand_choice = np.random.choice(train_size, batch_size, replace = False)
        all_train_index.append(rand_choice)
        folders = [train_gif_dirs[j] for j in rand_choice]

        gif_index = []
        for folder in folders:
            gifs = natsorted(os.listdir(folder))
            gif_choices = np.random.choice(range(len(gifs)), shots, replace = False)
            gif_name_choices = [os.path.join(folder, gifs[item]) for item in gif_choices]
            all_train_filenames.extend(gif_name_choices)
            gif_index.append(gif_choices)

        all_train_gif_index.append(gif_index)
        if (i != 0 and i % val_interval == 0):
            rand_choice = np.random.choice(val_size, batch_size, replace = False)
            all_val_index.append(rand_choice)
            folders = [val_gif_dirs[j] for j in rand_choice]

            gif_index = []
            for folder in folders:
                gifs = natsorted(os.listdir(folder))
                gif_choices = np.random.choice(range(len(gifs)), shots, replace = False)
                gif_name_choices = [os.path.join(folder, gifs[item]) for item in gif_choices]
                all_val_filenames.extend(gif_name_choices)
                gif_index.append(gif_choices)

            all_val_gif_index.append(gif_index)

    #print(all_train_filenames[0])
    #print(len(all_val_filenames))
    load_data_variables['all_train_filenames'] = all_train_filenames
    load_data_variables['all_val_filenames'] = all_val_filenames

    load_data_variables['all_train_index'] = all_train_index
    load_data_variables['all_val_index'] = all_val_index

    load_data_variables['all_train_gif_index'] = all_train_gif_index
    load_data_variables['all_val_gif_index'] = all_val_gif_index

    load_data_variables['train_itr'] = 0
    load_data_variables['val_itr'] = 0
        
    
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

def generate_training_batch(it):
    tot = load_data_config['tasks_per_batch'] * load_data_config['K-shots'] * 2
    itr = load_data_variables['train_itr']
    filenames = load_data_variables['all_train_filenames']
    '''img_listA = [read_gif(filenames[i]) for i in range(itr, itr + tot) if (i % 2 == 0)]
    img_listB = [read_gif(filenames[i]) for i in range(itr, itr + tot) if (i % 2 != 0)]
    load_data_variables['train_itr'] = itr + tot
    img_tensorsA = tf.stack(img_listA)
    img_tensorsB = tf.stack(img_listB)
    
    img_tensorsA = tf.reshape(img_tensorsA, (-1, load_data_constants['img_height'], \
                   load_data_constants['img_width'], \
                   load_data_constants['color_channels']))
    img_tensorsB = tf.reshape(img_tensorsB, (-1, load_data_constants['img_height'], \
                   load_data_constants['img_width'], \
                   load_data_constants['color_channels']))'''

    img_namesA = [filenames[i] for i in range(itr, itr + tot) if (i % 2 == 0)]
    img_namesB = [filenames[i] for i in range(itr, itr + tot) if (i % 2 != 0)]
    load_data_variables['train_itr'] = itr + tot

    all_train_index = load_data_variables['all_train_index']
    all_train_gif_index = load_data_variables['all_train_gif_index']

    index = all_train_index[it]
    gif_index = all_train_gif_index[it]

    size1 = len(index)
    size2 = len(gif_index[0])

    data = load_data_variables['train_data']

    stateA = [data[index[int(i/size2)]]['demoX'][gif_index[int(i/size2)][i%size2]]\
              for i in range(size1*size2) if (i % 2 == 0)]
    stateB = [data[index[int(i/size2)]]['demoX'][gif_index[int(i/size2)][i%size2]]\
              for i in range(size1*size2) if (i % 2 != 0)]
    
    actionA = [data[index[int(i/size2)]]['demoU'][gif_index[int(i/size2)][i%size2]]\
              for i in range(size1*size2) if (i % 2 == 0)]
    actionB = [data[index[int(i/size2)]]['demoU'][gif_index[int(i/size2)][i%size2]]\
              for i in range(size1*size2) if (i % 2 != 0)]
    return img_namesA, img_namesB, stateA, stateB, actionA, actionB

def generate_validation_batch(it):
    it = int(it / load_data_config['val_interval'])-1
    tot = load_data_config['tasks_per_batch'] * load_data_config['K-shots'] * 2
    itr = load_data_variables['val_itr']
    filenames = load_data_variables['all_val_filenames']
    '''img_listA = [read_gif(filenames[i]) for i in range(itr, itr + tot) if (i % 2 == 0)]
    img_listB = [read_gif(filenames[i]) for i in range(itr, itr + tot) if (i % 2 != 0)]
    load_data_variables['val_itr'] = itr + tot
    img_tensorsA = tf.stack(img_listA)
    img_tensorsB = tf.stack(img_listB)
    
    img_tensorsA = tf.reshape(img_tensorsA, (-1, load_data_constants['img_height'], \
                   load_data_constants['img_width'], \
                   load_data_constants['color_channels']))
    img_tensorsB = tf.reshape(img_tensorsB, (-1, load_data_constants['img_height'], \
                   load_data_constants['img_width'], \
                   load_data_constants['color_channels']))'''

    img_namesA = [filenames[i] for i in range(itr, itr + tot) if (i % 2 == 0)]
    img_namesB = [filenames[i] for i in range(itr, itr + tot) if (i % 2 != 0)]
    load_data_variables['val_itr'] = itr + tot
    #print(img_tensors)
    #tf.print(img_tensors[0][0][0][0], output_stream = sys.stdout)
    #tf.Print(img_tensors[0][0][0][0], [img_tensors[0][0][0][0]])

    #Well, I guess it's right here...
    #To be further checked
    #print(img_tensorsA)

    all_val_index = load_data_variables['all_val_index']
    all_val_gif_index = load_data_variables['all_val_gif_index']

    index = all_val_index[it]
    gif_index = all_val_gif_index[it]

    size1 = len(index)
    size2 = len(gif_index[0])
    
    data = load_data_variables['val_data']

    stateA = [data[index[int(i/size2)]]['demoX'][gif_index[int(i/size2)][i%size2]]\
              for i in range(size1*size2) if (i % 2 == 0)]
    stateB = [data[index[int(i/size2)]]['demoX'][gif_index[int(i/size2)][i%size2]]\
              for i in range(size1*size2) if (i % 2 != 0)]

    actionA = [data[index[int(i/size2)]]['demoU'][gif_index[int(i/size2)][i%size2]]\
              for i in range(size1*size2) if (i % 2 == 0)]
    actionB = [data[index[int(i/size2)]]['demoU'][gif_index[int(i/size2)][i%size2]]\
              for i in range(size1*size2) if (i % 2 != 0)]
    return img_namesA, img_namesB, stateA, stateB, actionA, actionB

def read_gif(filename):
    print(filename)
    img = tf.image.decode_gif(filename)
    img.set_shape((load_data_constants['frames_in_gif'], \
                   load_data_constants['img_height'], \
                   load_data_constants['img_width'], \
                   load_data_constants['color_channels']))
    img = tf.cast(img, tf.float32)
    img /= 255.0
    return img

def data_preload(config):
    init_load_data_config(config)
    load_states_and_actions()
    split_selected_data()
    load_gif_dirs()
    gen_batch_filenames()
