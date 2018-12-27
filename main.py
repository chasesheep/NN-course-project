from load_data import data_preload, generate_training_batch, \
    generate_validation_batch, load_data_variables
from mil import *

import os
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

config = {
    'learning_rate': 0.001,
    'meta_learning_rate': 0.001,
    'K-shots': 1,
    'train_samples': 500,
    'validate_samples': 100,
    'split_channels': 30,
    'kernel_size': 3,
    'fc_layer_size': 200,
    'two_head': False,
    'iterations': 1000,
    'val_interval': 50,
    'print_interval': 50,
    'maml_tasks_per_batch': 5,
    'decay': 0.9,
    'strides': [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]],
    'fc_bt': True,
    'clip_min': -10.0,
    'clip_max': 10.0,

    'checkpoint_num': 10,
    'log_dir': 'logs/'
}

constants = {
    'frames_in_gif': 50,
    'conv_layers': 3,
    'action_dim': 2
    # 'gif_width': 80,
    # 'gif_height': 64
}

if __name__ == '__main__':
    graph = tf.Graph()
    # GPU options: this part directly migrated from original settings
    print('Setting GPU options...')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(graph=graph, config=tf_config)

    # load data
    with sess.as_default():
        data_preload(config)

    init_network_config(config)
    init_network(graph, True)
    init_network(graph, False)

    with graph.as_default():
        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=config['checkpoint_num'])

    if not os.path.exists(config['log_dir']):
        os.mkdir(config['log_dir'])
    if tf.train.get_checkpoint_state(config['log_dir']):
        saver.restore(sess, tf.train.latest_checkpoint(config['log_dir']))
    else:
        with graph.as_default():
            init_op = tf.global_variables_initializer()
            sess.run(init_op, feed_dict=None)

    prelosses, postlosses = [], []
    train_prelosses_history, train_postlosses_history = [], []
    val_prelosses_history, val_postlosses_history = [], []
    for itr in range(config['iterations']):
        #print('Running itr '+str(itr))
        img_namesA, img_namesB, stateA, stateB, actionA, actionB = \
                generate_training_batch(itr)
        # imgA = imgA.eval(session=sess)
        #img_namesA = img_namesB = ['0.gif', '1.gif', '2.gif', '3.gif', '4.gif']
        #print(img_namesA)
        #print(img_namesB)
        feed_dict = {
            mil_variables['stateA']: stateA,
            mil_variables['actionA']: actionA,
            mil_variables['img_namesA']: img_namesA,
            mil_variables['stateB']: stateB,
            mil_variables['actionB']: actionB,
            mil_variables['img_namesB']: img_namesB
        }
        with graph.as_default():
            result = sess.run(
                [mil_variables['train_lossA'],
                 mil_variables['train_lossB']],
                feed_dict=feed_dict)
            #print(result)
            #prelosses.extend(np.mean(result[0]))
            #postlosses.extend(np.mean(result[1]))
            prelosses.append(result[0])
            postlosses.append(result[1])

        if itr % config['print_interval'] == 0 and itr != 0:
            preloss = np.mean(prelosses)
            postloss = np.mean(postlosses)
            train_prelosses_history.append(preloss)
            train_postlosses_history.append(postloss)
            prelosses, postlosses = [], []
            print('Train Iteration %d: \n'
                  'preloss is %.2f, postloss is %.2f' % (
                      itr, preloss, postloss))
            with graph.as_default():
                saver.save(sess, config['log_dir'] + '_%d' % itr)

        if itr % config['val_interval'] == 0 and itr != 0:
            imgA, imgB, stateA, stateB, actionA, actionB = \
                generate_validation_batch(itr)
            feed_dict = {
                mil_variables['stateA']: stateA,
                mil_variables['actionA']: actionA,
                mil_variables['img_namesA']: img_namesA,
                mil_variables['stateB']: stateB,
                mil_variables['actionB']: actionB,
                mil_variables['img_namesB']: img_namesB
            }
            with graph.as_default():
                result = sess.run(
                    [mil_variables['val_lossA'],
                     mil_variables['val_lossB']],
                    feed_dict=feed_dict)
                preloss = np.mean(result[0])
                postloss = np.mean(result[1])
                print('Val Iteration %d: \n'
                      'preloss is %.2f, postloss is %.2f' % (
                          itr, preloss, postloss))
                val_prelosses_history.append(preloss)
                val_postlosses_history.append(postloss)

    '''plt.plot(train_prelosses_history)
    plt.plot(train_postlosses_history)
    plt.legend(['pre', 'post'], loc='upper left')
    plt.xlabel('iter')
    plt.ylabel('train loss')
    plt.show()'''
