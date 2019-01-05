from load_data import *
from mil import *
#from eval_reaching import evaluate_vision_reach

import os
import tensorflow as tf
import numpy as np
#import gym

# import matplotlib.pyplot as plt

config = {
    'learning_rate': 0.001,
    'meta_learning_rate': 0.001,
    'K-shots': 1,
    'train_samples': 700,
    'validate_samples': 100,
    'split_channels': 30,
    'kernel_size': 3,
    'fc_layer_size': 200,
    'two_head': False, # some bugs here
    'iterations': 30000,
    'val_interval': 100,
    'print_interval': 100,
    'maml_tasks_per_batch': 4,
    'decay': 0.9,
    'strides': [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]],
    'fc_bt': True,
    'clip_min': -10.0,
    'clip_max': 10.0,

    'checkpoint_num': 10,
    'log_dir': 'logs/',
    'is_record_gif': True,
    'task_type': 'reaching',
    'task_state': 'training',
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=tf_config)

    # load data
    if config['task_state'] == 'training':
        with sess.as_default():
            data_preload(config)
    elif config['task_state'] == 'testing':
        generate_test_demos()
    else:
        print('Unknown operation. Aborted.')
        exit(1)

    init_network_config(config)

    if config['task_state'] == 'training':
        init_network(graph, True)
        init_network(graph, False, False)
    else:
        init_network(graph, False, True)

    with graph.as_default():
        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=config['checkpoint_num'])

    if not os.path.exists(config['log_dir']):
        os.mkdir(config['log_dir'])
    log_dir = os.path.join(config['log_dir'], config['task_type'])
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if tf.train.get_checkpoint_state(log_dir):
        saver.restore(sess, tf.train.latest_checkpoint(log_dir))
    else:
        with graph.as_default():
            init_op = tf.global_variables_initializer()
            sess.run(init_op, feed_dict=None)

    if config['task_state'] == 'training':
        prelosses, postlosses = [], []
        train_prelosses_history, train_postlosses_history = [], []
        val_prelosses_history, val_postlosses_history = [], []
        for itr in range(config['iterations']):
            # print('Running itr '+str(itr))
            img_namesA, img_namesB, stateA, stateB, actionA, actionB = \
                generate_training_batch(itr)
            # imgA = imgA.eval(session=sess)
            # img_namesA = img_namesB = ['0.gif', '1.gif', '2.gif', '3.gif', '4.gif']
            # print(img_namesA)
            # print(img_namesB)
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
                    [mil_variables['train_op'],
                     mil_variables['train_lossA'],
                     mil_variables['train_lossB'],
                     mil_variables['train_outputBs'],
                     mil_variables['actionB']],
                    feed_dict=feed_dict)
                prelosses.append(result[1])
                postlosses.append(result[2])


            if itr % config['print_interval'] == 0 and itr != 0:
                preloss = np.mean(prelosses)
                postloss = np.mean(postlosses)
                train_prelosses_history.append(preloss)
                train_postlosses_history.append(postloss)
                prelosses, postlosses = [], []
                print('Train Iteration %d: \n'
                      'preloss is %.2f, postloss is %.2f' % (
                          itr, preloss, postloss))

                exp_act = result[4]
                our_act = result[3]
                logfile = open('log.txt', 'a')
                logfile.write('---------------------------\n')
                logfile.write(str(itr)+'\n')
                logfile.write(str(result[2])+'\n')
                logfile.write('example loss: '+str(np.linalg.norm(exp_act[0] - our_act[0], 'fro'))+'\n')
                logfile.write('actions example\n')
                for item in range(50):
                    logfile.write(str(exp_act[0][item]) + ' ' + str(our_act[0][item])+'\n')

                logfile.close()
                with graph.as_default():
                    saver.save(sess, log_dir + ('/model_%d' % itr))

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
                         mil_variables['val_lossB'],
                         mil_variables['train_outputBs'],
                         mil_variables['actionB']
                         ],
                        feed_dict=feed_dict)
                    preloss = np.mean(result[0])
                    postloss = np.mean(result[1])

                    exp_val_act = result[3]
                    our_val_act = result[2]

                    vlogfile = open('val_log.txt', 'a')
                    vlogfile.write('---------------------------\n')
                    vlogfile.write(str(itr) + '\n')
                    vlogfile.write(str(result[1]) + '\n')
                    vlogfile.write('example loss: ' + str(np.linalg.norm(exp_val_act[0] - our_val_act[0], 'fro')) + '\n')
                    vlogfile.write('actions example\n')
                    for item in range(50):
                        vlogfile.write(str(exp_val_act[0][item]) + ' ' + str(our_val_act[0][item]) + '\n')

                    vlogfile.close()

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
    elif config['task_state'] == 'testing':
        print('Testing')
        env = gym.make('ReacherMILTest-v1')
        evaluate_vision_reach(env, graph, mil_variables, sess,
                              load_data_variables, config['is_record_gif'],
                              log_dir)
