import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os
from tqdm import tqdm

""" ==================== GLOBAL PARAMETERS ==================== """
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-agents", type=int, default=4, help="number of agents")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--buffer-size", type=int, default=20, help="size * batch_size = real_buffer_size")
    parser.add_argument("--write-summary-every", type=int, default=200, help="size * batch_size = real_buffer_size")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="exponential moving average ratio")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--alpha-init", type=float, default=1.0, help="init value of temperature")
    parser.add_argument("--alpha-decay", type=float, default=0.9999, help="temperature decay")
    parser.add_argument("--adv-eps", type=float, default=1e-3, help="adversarial training rate")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./sac_vs_sac/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-good", type=str, default="./sac_vs_sac/", help="which good policy to load")
    parser.add_argument("--load-bad", type=str, default="./sac_vs_sac/", help="which bad policy to load")
    # Evaluation
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()
""" =========================================================== """


def make_env(scenario_name, benchmark=False):
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
    return output


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
    # train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return gaussian


def build_multi_distribution(input_op, num_units, output_dim, output_num, scope='multi_dist', reuse=tf.AUTO_REUSE, getter=None):
    params = {'kernel_initializer': tf.orthogonal_initializer(), 'bias_initializer': tf.constant_initializer(0.0)}
    output_dists = []
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        l1 = tf.layers.dense(input_op, num_units, activation=tf.nn.relu, name='layer_1', **params)
        l2 = tf.layers.dense(l1, num_units, activation=tf.nn.relu, name='layer_2', **params)
        for idx in range(output_num):
            with tf.variable_scope('agent_{}'.format(idx)):
                mu = tf.layers.dense(l2, output_dim, activation=tf.tanh, use_bias=True, name='mu', **params)
                sigma = tf.layers.dense(l2, output_dim, activation=tf.nn.softplus, use_bias=True, name='sigma', **params)
                gaussian = tf.distributions.Normal(mu, sigma)
                output_dists.append(gaussian)
    return output_dists


def initialize_multi_replay_buffer(env, sess, buffers, agents, episode_len=50):
    """initialize_multi_replay_buffer
    buffers: type list
    agents: type list
    """
    print(' [*] Initializing replay buffers...')
    pbar = tqdm(total=buffers[0].capacity)
    while buffers[0].n_sample <= buffers[0].capacity:
        done = False
        episode_len_cnt = 0
        global_s = env.reset()
        while episode_len_cnt <= episode_len and not done:
            global_action = [agent.choose_action(sess, s) for agent, s in zip(agents, global_s)]
            global_action_array = np.concatenate([np.expand_dims(a, axis=1) for a in global_action], axis=-1)
            global_s_next, global_reward, global_done, info = env.step(global_action)
            done = any(global_done)
            episode_len_cnt += 1
            if done or episode_len_cnt > episode_len and not done:
                break
            for bf, s, r, s_next in zip(buffers, global_s, global_reward, global_s_next):
                bf.restore(s, global_action_array, r, s_next)
            global_s = global_s_next
        pbar.update(episode_len_cnt)
    print(' [*] All agent buffer initialized')
    pbar.close()


def save_state(fname, global_step, saver=None):
    """Save all the variables in the current session to the location <fname>"""
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if saver is None:
        saver = tf.train.Saver()
    saver.save(sess, fname + '/model', global_step=global_step)
    return saver

def load_state_v2(model_path, saver=None):
    """ my implementation, model_path could be a directory/folder """
    import re
    print(" [*] Reading checkpoints in {}...".format(model_path))
    if saver is None:
        saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_path, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint in {}".format(model_path))
        return False, 0

def show_all_variables(scope=None):
    all_variables = tf.trainable_variables(scope=scope)
    slim.model_analyzer.analyze_vars(all_variables, print_info=True)


class EMAGetter(object):
    def __init__(self, tau):
        self.ema = tf.train.ExponentialMovingAverage(decay=1.0 - tau)

    def __call__(self, getter, name, *args, **kwargs):
        return self.ema.average(getter(name, *args, **kwargs))


class SoftActorCritic(object):
    def __init__(self, s_dim, a_dim, scope, arglist, agent_index=0):
        # placeholders
        self.state = tf.placeholder(tf.float32, [None, s_dim], name=scope + '_state')
        self.action = tf.placeholder(tf.float32, [None, a_dim, arglist.num_agents], name=scope + '_action')
        self.reward = tf.placeholder(tf.float32, [None, 1], name=scope + '_reward')
        self.s_next = tf.placeholder(tf.float32, [None, s_dim], name=scope + '_s_next')
        self.alpha = tf.placeholder(tf.float32, name='alpha')
        each_agent_actions = tf.unstack(self.action, axis=-1)
        this_agent_action = each_agent_actions[agent_index]
        joint_action_reshaped = tf.reshape(self.action, [-1, a_dim * arglist.num_agents])

        # build target networks
        with tf.variable_scope(scope):
            # each_agent_policies = build_multi_distribution(self.state, 64, a_dim, arglist.num_agents, scope='policy')
            self.policy = build_distribution(self.state, 64, a_dim, scope='policy')
            self.execution = tf.clip_by_value(self.policy.sample(), -1.5, 1.5)
            s_concat_a = tf.concat([self.state, joint_action_reshaped], axis=-1)
            self.target_q = build_mlp(s_concat_a, 64, 1, scope='q')
            self.value = build_mlp(self.state, 64, 1, scope='value')
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/q')
        v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/value')
        pi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/policy')
        # print(' [*] Network parameter number: Q: {}, V: {}, pi: {}'.format(len(q_vars), len(v_vars), len(pi_vars)))

        # exp moving average only used for value params
        with tf.variable_scope(scope):
            self.ema_getter = EMAGetter(tau=arglist.tau)
            target_update = [self.ema_getter.ema.apply(v_vars)]
            self.v_next = build_mlp(self.s_next, 64, 1, scope='value', reuse=True, getter=self.ema_getter)

        # losses
        with tf.control_dependencies(target_update):
            target_next_value = self.reward + arglist.gamma * self.v_next
            target_next_value = tf.stop_gradient(target_next_value)
            self.q_loss = tf.losses.mean_squared_error(labels=target_next_value, predictions=self.target_q)
            self.q_optim = tf.train.AdamOptimizer(arglist.lr).minimize(self.q_loss, var_list=q_vars)

            log_prob = self.policy.log_prob(this_agent_action)
            estimate_value = tf.stop_gradient(self.target_q - self.alpha * log_prob)
            self.v_loss = tf.losses.mean_squared_error(labels=estimate_value, predictions=self.value)
            self.v_optim = tf.train.AdamOptimizer(arglist.lr).minimize(self.v_loss, var_list=v_vars)

            fake_reward = tf.stop_gradient(self.alpha * log_prob - self.target_q + self.value)
            self.pi_loss = tf.reduce_mean(tf.reduce_sum(fake_reward * log_prob, axis=-1))
            self.pi_optim = tf.train.AdamOptimizer(arglist.lr).minimize(self.pi_loss, var_list=pi_vars)

            # for idx in range(arglist.num_agents):
            #     if idx == agent_index:
            #         continue
            #     mle = each_agent_policies[idx].log_prob(each_agent_actions[idx])
            #     other_agent_mle.append(-tf.reduce_mean(mle))
            # self.mle_loss = tf.add_n(other_agent_mle)
            # self.mle_optim = tf.train.AdamOptimizer(arglist.lr).minimize(self.mle_loss, var_list=pi_vars)

        self.train_vars = q_vars + v_vars + pi_vars
        self.replay_buffer = ReplayBuffer(arglist.buffer_size * arglist.batch_size, s_dim, a_dim, arglist.num_agents)
        self.summaries = tf.summary.merge([
            tf.summary.scalar('agent_{}/pi_loss'.format(agent_index), self.pi_loss),
            tf.summary.scalar('agent_{}/v_loss'.format(agent_index), self.v_loss),
            tf.summary.scalar('agent_{}/q_loss'.format(agent_index), self.q_loss),
            tf.summary.scalar('agent_{}/mean_reward'.format(agent_index), tf.reduce_mean(self.reward))
        ])
        print(' [*] SAC Agent {} built finished...'.format(scope))

    def choose_action(self, sess, s):
        if len(s.shape) < 2:
            s = s[None, :]
        return sess.run(self.execution, feed_dict={self.state: s})[0]

    def update(self, sess, alpha=1.0, batch_size=1024):
        s, a, r, s_next = self.replay_buffer.sample(batch_size)
        feed_dict = {self.state: s, self.action: a, self.reward: r, self.s_next: s_next, self.alpha: alpha}
        optims = [self.pi_optim, self.q_optim, self.v_optim]
        sess.run(optims, feed_dict=feed_dict)
        sess.run(optims, feed_dict=feed_dict)

    def collect_summary(self, sess, writer, global_step, alpha=1.0):
        s, a, r, s_next = self.replay_buffer.sample(1024)
        feed_dict = {self.state: s, self.action: a, self.reward: r, self.s_next: s_next, self.alpha: alpha}
        sumstr = sess.run(self.summaries, feed_dict=feed_dict)
        writer.add_summary(sumstr, global_step=global_step)
    

class ReplayBuffer(object):
    """ReplayBuffer
        NOTE: The action defined in multi-agent setting here refers to the GLOBAL JOINT action
    """
    def __init__(self, size, s_dim, a_dim, n_agents):
        self.buffer_state = np.zeros((size, s_dim), dtype=np.float32)
        self.buffer_action = np.zeros((size, a_dim, n_agents), dtype=np.float32)
        self.buffer_reward = np.zeros((size, 1), dtype=np.float32)
        self.buffer_s_next = np.zeros((size, s_dim), dtype=np.float32)
        self.capacity = size
        self.ptr = 0
        self.n_sample = 0

    def sample(self, batch_size):
        rand_indices = np.random.randint(0, self.capacity, size=batch_size)
        batch_state = self.buffer_state[rand_indices, :]
        batch_action = self.buffer_action[rand_indices, :, :]
        batch_reward = self.buffer_reward[rand_indices, :]
        batch_s_next = self.buffer_s_next[rand_indices, :]
        return batch_state, batch_action, batch_reward, batch_s_next

    def restore(self, s, a, r, s_next):
        self.buffer_state[self.ptr, :] = s
        self.buffer_action[self.ptr, :, :] = a
        self.buffer_reward[self.ptr, :] = r
        self.buffer_s_next[self.ptr, :] = s_next
        self.ptr = (self.ptr + 1) % self.capacity
        self.n_sample += 1
        return self.ptr


if __name__ == '__main__':
    arglist = parse_args()
    env = make_env(arglist.scenario)
    print(' [*] Number of agents: {}'.format(env.n))
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    global_agents = []
    global_s_dim = [space.shape[0] for space in env.observation_space]
    global_a_dim = [space.n for space in env.action_space]
    print(' [*] s_dim: {}, a_dim: {}'.format(global_s_dim, global_a_dim))
    good_vars, bad_vars = [], []
    for index in range(env.n):
        agent = SoftActorCritic(s_dim=global_s_dim[index], 
                                a_dim=global_a_dim[index],
                                scope='sac_agent_{}'.format(index),
                                arglist=arglist,
                                agent_index=index)
        global_agents.append(agent)
        if index == env.n - 1:
            bad_vars += agent.train_vars
        else:
            good_vars += agent.train_vars
    global_alpha = arglist.alpha_init

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    # Load adversarial states separately
    global_step = 0
    episode_counter = 0
    if arglist.load_good:
        good_saver = tf.train.Saver(good_vars)
        _, global_step = load_state_v2(arglist.load_good, good_saver)
    if arglist.load_bad:
        bad_saver = tf.train.Saver(bad_vars, bad_vars)
        _, global_step = load_state_v2(arglist.load_bad, bad_saver)
    global_alpha *= (arglist.alpha_decay ** global_step)
    # initialize all buffers
    end_step = global_step + arglist.num_episodes
    global_buffers = [agent.replay_buffer for agent in global_agents]
    initialize_multi_replay_buffer(env, sess, global_buffers, global_agents, episode_len=arglist.max_episode_len)

    # Initialize writer, global_step, global_saver, env
    global_saver = tf.train.Saver()
    global_writer = tf.summary.FileWriter(arglist.save_dir, sess.graph)
    global_state = env.reset()
    reward_list = [[] for _ in range(env.n)]
    total_reward = [0.0 for _ in range(env.n)]
    show_all_variables()
    print(' [*] Initialization ready... Start to train from global_step {}...'.format(global_step))

    while global_step < end_step:
        global_action = [agent.choose_action(sess, s) for agent, s in zip(global_agents, global_state)]
        global_action_array = np.concatenate([np.expand_dims(a, axis=1) for a in global_action], axis=-1)
        global_s_next, global_reward, global_done, info = env.step(global_action)
        done = any(global_done)
        if done or episode_counter > arglist.max_episode_len:
            global_state = env.reset()
            episode_counter = 0
            for rll, trr in zip(reward_list, total_reward):
                rll.append(trr)
            total_reward = [0.0 for _ in range(env.n)]
            continue

        for bf, s, r, s_next in zip(global_buffers, global_state, global_reward, global_s_next):
            bf.restore(s, global_action_array, r, s_next)
        for idx, r in enumerate(global_reward):
            total_reward[idx] += r

        if not arglist.test:
            for idx, agent in enumerate(global_agents):
                agent.update(sess, alpha=global_alpha, batch_size=arglist.batch_size)
                if global_step % arglist.write_summary_every == 1:
                    agent.collect_summary(sess, global_writer, global_step=global_step, alpha=global_alpha)
                    temp_sumstr = tf.Summary(value=[tf.Summary.Value(tag='agent_{}/alpha'.format(idx), simple_value=global_alpha)])
                    global_writer.add_summary(temp_sumstr, global_step=global_step)
                if global_step % arglist.save_rate == arglist.save_rate - 1:
                    save_state(arglist.save_dir, global_step=global_step)
            env.render()
        # else:
        #     env.render()

        global_state = global_s_next
        global_alpha *= arglist.alpha_decay
        global_step += 1
        episode_counter += 1
    print(' [*] Training finished!!')

    plt.figure()
    plt.plot(reward_list[0], linewidth=1.5)
    plt.plot(reward_list[-1], linewidth=1.5)
    plt.title('Total reward with episode length = {}'.format(arglist.max_episode_len))
    plt.legend(['adversaries', 'good policy'], loc='best')
    plt.show()

    reward_array = np.array(reward_list, dtype=np.float32)
    saved_filename = './benchmark_files/reward_result.npy'
    np.save(saved_filename, reward_array)
    print(' [*] reward file saved in {}'.format(saved_filename))