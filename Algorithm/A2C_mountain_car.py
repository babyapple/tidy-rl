import gym
import numpy as np
import tensorflow as tf


class Memory(object):
    def __init__(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []

    def store_transition(self, obs0, act, rwd):
        self.ep_obs.append(obs0)
        self.ep_act.append(act)
        self.ep_rwd.append(rwd)

    def covert_to_array(self):
        array_obs = np.vstack(self.ep_obs)
        array_act = np.vstack(self.ep_act)
        array_rwd = np.array(self.ep_rwd)
        return array_obs, array_act, array_rwd

    def reset(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []


class ActorNetwork(object):
    def __init__(self, act_dim, name):
        self.act_dim = act_dim
        self.name = name

    def step(self, obs, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h1 = tf.layers.dense(obs, 64, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu)
            mu = 2 * tf.layers.dense(h2, self.act_dim, activation=tf.nn.tanh)
            sigma = tf.layers.dense(h2, self.act_dim, activation=tf.nn.softplus)
            pd = tf.distributions.Normal(loc=mu, scale=sigma)
        return pd

    def choose_action(self, obs, reuse=False):
        pd = self.step(obs, reuse)
        action = tf.squeeze(pd.sample(1), axis=0)
        action = tf.clip_by_value(action, -1.0, 1.0)
        return action

    def get_neglogp(self, obs, act, reuse=True):
        pd = self.step(obs, reuse)
        neglogp = pd.prob(act)
        return neglogp


class ValueNetwork(object):
    def __init__(self, name):
        self.name = name

    def step(self, obs, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h1 = tf.layers.dense(obs, 64, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu)
            value = tf.layers.dense(inputs=h2, units=1)
            return value

    def get_value(self, obs, reuse=False):
        value = self.step(obs, reuse)
        return value


class ActorCritic:
    def __init__(self, act_dim, obs_dim, lr_actor, lr_value, gamma):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr_actor = lr_actor
        self.lr_value = lr_value
        self.gamma = gamma

        self.OBS = tf.placeholder(tf.float32, [None, self.obs_dim], name="observation")
        self.ACT = tf.placeholder(tf.float32, [None, self.act_dim], name="action")
        self.Q_VAL = tf.placeholder(tf.float32, [None, 1], name="q_value")

        actor = ActorNetwork(self.act_dim, 'actor')
        critic = ValueNetwork('critic')
        self.memory = Memory()

        self.action = actor.choose_action(self.OBS)
        neglogp = actor.get_neglogp(self.OBS, self.ACT)
        self.value = critic.get_value(self.OBS)
        self.advantage = self.Q_VAL - self.value

        actor_loss = tf.reduce_mean(neglogp * self.advantage)
        self.actor_train_op = tf.train.AdamOptimizer(self.lr_actor).minimize(actor_loss)

        value_loss = tf.reduce_mean(tf.square(self.advantage))
        self.value_train_op = tf.train.AdamOptimizer(self.lr_value).minimize(value_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def step(self, obs):
        if obs.ndim < 2: obs = obs[np.newaxis, :]
        action = self.sess.run(self.action, feed_dict={self.OBS: obs})
        action = np.squeeze(action, 1).clip(action, env.action_space.low[0], env.action_space.high[0])

        value = self.sess.run(self.value, feed_dict={self.OBS: obs})
        return action, value

    def learn(self, last_value, done):
        obs, act, rwd = self.memory.covert_to_array()

        q_value = self.compute_q_value(last_value, done, rwd)

        self.sess.run(self.actor_train_op, {self.OBS: obs, self.ACT: act, self.Q_VAL: q_value})
        self.sess.run(self.value_train_op, {self.OBS: obs, self.Q_VAL: q_value})

        self.memory.reset()

    def compute_q_value(self, last_value, done, rwd):
        q_value = np.zeros_like(rwd)
        value = 0 if done else last_value
        for t in reversed(range(0, len(rwd))):
            value = value * self.gamma + rwd[t]
            q_value[t] = value
        return q_value[:, np.newaxis]


env = gym.make('MountainCarContinuous-v0')
env.seed(1)
env = env.unwrapped

agent = ActorCritic(act_dim=env.action_space.shape[0], obs_dim=env.observation_space.shape[0],
                    lr_actor=0.0001, lr_value=0.0002, gamma=0.99)

nepisode = 1000
nstep = 200

for i_episode in range(nepisode):
    obs0 = env.reset()
    ep_rwd = 0

    for t in range(nstep):
        act, _ = agent.step(obs0)

        obs1, rwd, done, info = env.step(act)

        agent.memory.store_transition(obs0, act, rwd)
        ep_rwd += rwd

        obs0 = obs1

        if t == nstep - 1:
            _, last_value = agent.step(obs1)
            agent.learn(last_value, done)

    print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)