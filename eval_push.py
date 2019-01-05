import glob
import os
import numpy as np
import tensorflow as tf
import pickle
import imageio
from PIL import Image
import errno

XML_PATH = 'sim_push_xmls/'
SCALE_FILE_PATH = 'data/scale_and_bias_sim_push.pkl'
from gym.envs.mujoco.pusher import PusherEnv
from eval_reaching import load_scale_and_bias
from sampler_utils import rollout


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class TFAgent(object):
    def __init__(self, mil_variables, scale_bias_file, sess):
        self.sess = sess
        self.mil_variables = mil_variables

        if scale_bias_file:
            self.scale, self.bias = load_scale_and_bias(scale_bias_file)
        else:
            self.scale = None

    def set_demo(self, demo_gif, demoX, demoU):
        demo_gif = np.array(demo_gif)
        N, T, H, W, C = demo_gif.shape
        self.update_batch_size = N
        self.T = T
        self.demo_gif = demo_gif
        self.demoX = demoX
        self.demoU = demoU

    def get_vision_action(self, image, obs, t=-1):

        image = np.expand_dims(image, 0).transpose(0, 3, 2, 1).astype('float32') / 255.0
        image = image.reshape((1, 1, -1))

        obs = obs.reshape((1, 1, 20))

        if self.scale is not None:
            feed_dict = {
                self.mil_variables['stateA']: self.demoX.dot(self.scale) + self.bias,
                self.mil_variables['actionA']: self.demoU,
                self.mil_variables['stateB']: obs.dot(self.scale) + self.bias,
                self.mil_variables['actionB']: self.demoU,
                self.mil_variables['img_namesA']: self.demo_gif,
                self.mil_variables['obs']: image
            }
            action = self.sess.run(self.mil_variables['val_op'], feed_dict=feed_dict)
        else:
            raise NotImplementedError("scale should not be None")

        return action, dict()


def load_demo(task_id, demo_dir, demo_inds):
    demo_info = pickle.load(open(demo_dir + task_id + '.pkl', 'rb'))
    demoX = demo_info['demoX'][demo_inds, :, :]
    demoU = demo_info['demoU'][demo_inds, :, :]
    d1, d2, _ = demoX.shape
    demoX = np.reshape(demoX, [1, d1 * d2, -1])
    demoU = np.reshape(demoU, [1, d1 * d2, -1])

    # read in demo video
    demo_gifs = [(demo_dir + 'object_' + task_id + '/cond%d.samp0.gif' % demo_ind) for demo_ind in demo_inds]

    return demoX, demoU, demo_gifs, demo_info


def load_env(demo_info):
    xml_filepath = demo_info['xml']
    suffix = xml_filepath[xml_filepath.index('pusher'):]
    prefix = XML_PATH + 'test2_ensure_woodtable_distractor_'
    xml_filepath = str(prefix + suffix)
    print(xml_filepath)

    env = PusherEnv(**{'xml_file': xml_filepath, 'distractors': True})
    return env


def eval_success(path):
    obs = path['observations']
    target = obs[:, -3:-1]
    obj = obs[:, -6:-4]
    dists = np.sum((target - obj) ** 2, 1)  # distances at each timestep
    return np.sum(dists < 0.017) >= 10


def evaluate_push(sess, mil_variables, log_dir, demo_dir, id=-1, save_video=True, num_input_demos=1):
    scale_file = SCALE_FILE_PATH
    files = glob.glob(os.path.join(demo_dir, '*.pkl'))
    task_ids = [int(f.split('/')[-1][:-4]) for f in files]
    task_ids.sort()
    num_success = 0
    num_trials = 0
    trials_per_task = 5

    for task_id in task_ids:
        demo_inds = [1]
        if num_input_demos > 1:
            demo_inds += range(12, 12 + int(num_input_demos / 2))
            demo_inds += range(2, 2 + int((num_input_demos - 1) / 2))
        assert len(demo_inds) == num_input_demos

        demoX, demoU, demo_gifs, demo_info = load_demo(str(task_id), demo_dir, demo_inds)

        env = load_env(demo_info)

        policy = TFAgent(mil_variables, scale_file, sess)
        policy.set_demo(demo_gifs, demoX, demoU)
        returns = []
        gif_dir = log_dir + '/evaluated_gifs/'
        mkdir_p(gif_dir)

        while True:
            video_suffix = gif_dir + str(id) + 'demo_' + str(num_input_demos) + '_' + str(len(returns)) + '.gif'
            path = rollout(env, policy, max_path_length=100, env_reset=True,
                           animated=True, speedup=1, always_return_paths=True, save_video=save_video, video_filename=video_suffix)
            num_trials += 1
            if eval_success(path):
                num_success += 1
            print('Return: ' + str(path['rewards'].sum()))
            returns.append(path['rewards'].sum())
            print('Average Return so far: ' + str(np.mean(returns)))
            print('Success Rate so far: ' + str(float(num_success) / num_trials))
            if len(returns) > trials_per_task:
                break

        tf.reset_default_graph()

    success_rate_msg = "Final success rate is %.5f" % (float(num_success) / num_trials)
    with open('logs/log_sim_push.txt', 'a') as f:
        f.write(success_rate_msg + '\n')
