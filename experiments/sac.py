import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

""" ==================== GLOBAL PARAMETERS ==================== """
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=20000, help="number of episodes")
    parser.add_argument("--buffer-size", type=int, default=20, help="size * batch_size = real_buffer_size")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="exponential moving average ratio")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--alpha-init", type=float, default=1.0, help="init value of temperature")
    parser.add_argument("--alpha-decay", type=float, default=0.9999, help="temperature decay")
    parser.add_argument("--adv-eps", type=float, default=1e-3, help="adversarial training rate")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-name", type=str, default="", help="name of which training state and model are loaded, leave blank to load seperately")
    parser.add_argument("--load-good", type=str, default="./maddpg_vs_maddpg/", help="which good policy to load")
    parser.add_argument("--load-bad", type=str, default="./maddpg_vs_maddpg/", help="which bad policy to load")
    # Evaluation
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()
""" ========================================================== """


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def build_mlp(input_op, num_units, output_dim, num_layers=3, scope='mlp', reuse=tf.AUTO_REUSE, getter=None):
    output = input_op
    params = {'kernel_initializer': tf.orthogonal_initializer(), 'bias_initializer': tf.constant_initializer(0.0)}
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        for idx in range(num_layers - 1):
            output_name = 'layer_{}'.format(idx)
            output = tf.layers.dense(output, num_units, activation=tf.nn.relu, name=output_name, **params)
        output = tf.layers.dense(output, output_dim, activation=None, use_bias=True, name='output', **params)
    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return output, train_vars


def build_distribution(input_op, num_units, output_dim, num_layers=3, scope='dist', reuse=tf.AUTO_REUSE, getter=None):
    output = input_op
    params = {'kernel_initializer': tf.orthogonal_initializer(), 'bias_initializer': tf.constant_initializer(0.0)}
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        for idx in range(num_layers - 1):
            output_name = 'layer_{}'.format(idx)
            output = tf.layers.dense(output, num_units, activation=tf.nn.relu, name=output_name, **params)
        mu = tf.layers.dense(output, output_dim, activation=tf.tanh, use_bias=True, name='mu', **params)
        sigma = tf.layers.dense(output, output_dim, activation=tf.nn.softplus, use_bias=True, name='sigma', **params)
        gaussian = tf.distributions.Normal(mu, sigma)
    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return gaussian, train_vars


class EMAGetter(object):
    def __init__(self, tau):
        self.ema = tf.train.ExponentialMovingAverage(decay=1.0 - tau)

    def __call__(self, getter, name, *args, **kwargs):
        return self.ema.average(getter(name, *args, **kwargs))


class SoftActorCritic(object):
    def __init__(self, s_dim, a_dim, scope, arglist):
        # placeholders
        self.state = tf.placeholder(tf.float32, [None, s_dim], name=scope + '_state')
        self.action = tf.placeholder(tf.float32, [None, a_dim], name=scope + '_action')
        self.reward = tf.placeholder(tf.float32, [None, 1], name=scope + '_reward')
        self.s_next = tf.placeholder(tf.float32, [None, s_dim], name=scope + '_s_next')
        self.alpha = tf.placeholder(tf.float32, name='alpha')

        # build target networks
        with tf.variable_scope(scope):
            self.policy, p_vars = build_distribution(self.state, 64, a_dim, scope='policy')
            self.execution = self.policy.sample(1)
            s_concat_a = tf.concat([self.state, self.action], axis=-1)
            self.target_q, q_vars = build_mlp(s_concat_a, 64, 1, scope='q')
            self.value, v_vars = build_mlp(self.state, 64, 1, scope='value')

        # exp moving average only used for value params
        with tf.variable_scope(scope):
            self.ema_getter = EMAGetter(tau=arglist.tau)
            target_update = [self.ema_getter.ema.apply(v_vars)]
            self.v_next, _ = build_mlp(self.s_next, 64, 1, scope='value', reuse=True, getter=self.ema_getter)

        # losses
        with tf.control_dependencies(target_update):
            target_next_value = self.reward + arglist.gamma * self.v_next
            target_next_value = tf.stop_gradient(target_next_value)
            self.q_loss = tf.losses.mean_squared_error(labels=target_next_value, predictions=self.target_q)

            log_prob = self.policy.log_prob(self.action)
            estimate_value = tf.stop_gradient(self.target_q - self.alpha * log_prob)
            self.v_loss = tf.losses.mean_squared_error(labels=estimate_value, predictions=self.value)

            fake_reward = tf.stop_gradient(self.alpha * log_prob - self.target_q + self.value)
            self.pi_loss = tf.reduce_mean(tf.reduce_sum(fake_reward * log_prob, axis=-1))

    def choose_action(self, sess, s):
        return sess.run(self.execution, feed_dict={self.state: s})

    def update(self, sess, s, a, r, s_next, alpha=1.0):
        feed_dict = {self.state: s, self.action: a, self.reward: r, self.s_next: s_next, self.alpha: alpha}


    

class ReplayBuffer(object):
    """ReplayBuffer
        NOTE: The action defined in multi-agent setting here refers to the GLOBAL JOINT action
    """
    def __init__(self, size, s_dim, a_dim):
        self.buffer_state = np.zeros((size, s_dim), dtype=np.float32)
        self.buffer_action = np.zeros((size, a_dim), dtype=np.float32)
        self.buffer_reward = np.zeros((size, 1), dtype=np.float32)
        self.buffer_s_next = np.zeros((size, s_dim), dtype=np.float32)
        self.capacity = size
        self.ptr = 0

    def sample(self, batch_size):
        rand_indices = np.random.randint(0, self.capacity, size=batch_size)
        batch_state = self.buffer_state[rand_indices, :]
        batch_action = self.buffer_action[rand_indices, :]
        batch_reward = self.buffer_reward[rand_indices, :]
        batch_s_next = self.buffer_s_next[rand_indices, :]
        return batch_state, batch_action, batch_reward, batch_s_next

    def restore(self, s, a, r, s_next):
        self.buffer_state[self.ptr, :] = s
        self.buffer_action[self.ptr, :] = a
        self.buffer_reward[self.ptr, :] = r
        self.buffer_s_next[self.ptr, :] = s_next
        self.ptr = (self.ptr + 1) % self.capacity
        return self.ptr



if __name__ == '__main__':
    pass