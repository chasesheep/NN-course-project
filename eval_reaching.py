import gym
import numpy as np
import tensorflow as tf

try:
    import cPickle as pickle
except:
    import pickle
import os
import time
import imageio

REACH_SUCCESS_THRESH = 0.05
REACH_SUCCESS_TIME_RANGE = 10
REACH_DEMO_CONDITIONS = 10
REACH_DEMO_LENGTH = 50


def load_scale_and_bias(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        scale = data['scale']
        scale = np.diag(1.0 / scale)
        bias = data['bias']
    return scale, bias


def evaluate_vision_reach(env, graph, mil_variables, sess, load_data_variables,
                          record_gifs, log_dir):
    demos = load_data_variables['demos']
    scale, bias = load_scale_and_bias('data/scale_and_bias_reaching.pkl')
    successes = []
    selected_demo = load_data_variables['test_select_demos']
    if record_gifs:
        record_gifs_dir = os.path.join(log_dir, 'evaluated_gifs')
        if not os.path.exists(record_gifs_dir):
            os.mkdir(record_gifs_dir)
    for i in range(len(selected_demo['selected_demoX'])):
        selected_demoO = selected_demo['selected_demoO'][i]
        selected_demoX = selected_demo['selected_demoX'][i]
        selected_demoU = selected_demo['selected_demoU'][i]
        if record_gifs:
            gifs_dir = os.path.join(record_gifs_dir, 'color_%d' % i)
            if not os.path.exists(gifs_dir):
                os.mkdir(gifs_dir)
        for j in range(REACH_DEMO_CONDITIONS):
            if j in demos[i]['demoConditions']:
                dists = []
                ob = env.reset()
                # use env.set_state here to arrange blocks
                Os = []
                log_path = os.path.join(gifs_dir, 'action_%d.txt' % j)
                with open(log_path, "wb") as f:
                    for t in range(REACH_DEMO_LENGTH):
                        # import pdb; pdb.set_trace()
                        env.render()
                        time.sleep(0.05)
                        obs, state = env.env.get_current_image_obs()
                        Os.append(obs)
                        # obs = np.transpose(obs, [2, 1, 0]) / 255.0
                        # obs = obs.reshape(1, 64, 80, 3)
                        obs = (obs / 255.0)[np.newaxis, np.newaxis, :, :, :]
                        # print(obs.shape)
                        state = state.reshape(1, 1, -1)

                        # print(selected_demoO)
                        # selected_demoX = np.reshape(selected_demoX, (-1, 10))
                        # selected_demoX = np.matmul(selected_demoX, scale) - bias
                        # selected_demoX = np.reshape(selected_demoX,
                        #                            (-1, 50, 10))
                        feed_dict = {
                            mil_variables['stateA']: selected_demoX.dot(
                                scale) - bias,
                            mil_variables['actionA']: selected_demoU,
                            mil_variables['img_namesA']: selected_demoO,
                            mil_variables['stateB']: state.dot(scale) - bias,
                            mil_variables['actionB']: selected_demoU,
                            mil_variables['obs']: obs
                        }

                        with graph.as_default():
                            action = sess.run(mil_variables['val_op'],
                                              feed_dict=feed_dict)
                            # print(action)
                            f.write(str(action) + '\n')

                        ob, reward, done, reward_dict = env.step(
                            np.squeeze(action))
                        dist = -reward_dict['reward_dist']
                        if t >= REACH_DEMO_LENGTH - REACH_SUCCESS_TIME_RANGE:
                            dists.append(dist)
                    f.close()
                if np.amin(dists) <= REACH_SUCCESS_THRESH:
                    successes.append(1.)
                else:
                    successes.append(0.)
                if record_gifs:
                    video = np.array(Os)
                    record_gif_path = os.path.join(gifs_dir,
                                                   'cond%d.samp0.gif' % j)
                    print('Saving gif sample to :%s' % record_gif_path)
                    imageio.mimwrite(record_gif_path, video)
            env.render(close=True)
            if j != REACH_DEMO_CONDITIONS - 1 or i != len(
                    selected_demo['selected_demoX']) - 1:
                env.env.next()
                env.render()
                time.sleep(0.5)
        print("Task %d: current success rate is %.5f" % (
            i, np.mean(successes)))
    success_rate_msg = "Final success rate is %.5f" % (np.mean(successes))
    print(success_rate_msg)
    with open('logs/log_sim_vision_reach.txt', 'a') as f:
        f.write(success_rate_msg + '\n')
