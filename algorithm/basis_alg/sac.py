import numpy as np
import tensorflow as tf
from utils.tf_utils import get_vars, RandomNormal
from .ddpg import DDPG

class SAC(DDPG):
    def __init__(self, args):
        self.args = args
        self.create_model()

        self.train_info_pi = {
            'Pi_loss': self.pi_loss
        }
        self.train_info_q = {
            'Q_loss': self.q_loss
        }
        self.train_info = {**self.train_info_pi, **self.train_info_q}

        self.step_info = {
            'Q_average': self.q_pi,
            'Pi_step_std': self.pi.std
        }

    def create_inputs(self):
        super().create_inputs()

        self.pi_noise_ph = tf.placeholder(tf.float32, [None]+self.args.acts_dims)
        self.pi_next_noise_ph = tf.placeholder(tf.float32, [None]+self.args.acts_dims)

    def create_network(self):
        def mlp_policy(obs_ph):
            with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                pi_dense1 = tf.layers.dense(obs_ph, 256, activation=tf.nn.relu, name='pi_dense1')
                pi_dense2 = tf.layers.dense(pi_dense1, 256, activation=tf.nn.relu, name='pi_dense2')
                pi_mean_logstd = tf.layers.dense(pi_dense2, self.args.acts_dims[0]*2, name='pi')
                pi_mean = pi_mean_logstd[:,:self.args.acts_dims[0]]
                pi_logstd = tf.clip_by_value(pi_mean_logstd[:,self.args.acts_dims[0]:], -20.0, 2.0)
            return RandomNormal(mean=pi_mean, logstd=pi_logstd)

        def mlp_q_value(obs_ph, acts_ph):
            state_ph = tf.concat([obs_ph, acts_ph], axis=1)
            with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                q_dense1 = tf.layers.dense(state_ph, 256, activation=tf.nn.relu, name='q_dense1')
                q_dense2 = tf.layers.dense(q_dense1, 256, activation=tf.nn.relu, name='q_dense2')
                q = tf.layers.dense(q_dense2, 1, name='q')
            return q

        with tf.variable_scope('main'):
            with tf.variable_scope('alpha'):
                self.log_alpha = tf.Variable(np.log(self.args.alpha_init), name='log_alpha', dtype=np.float32, trainable=True)
                self.alpha = tf.exp(self.log_alpha)

            def get_pi_log_p(pi, pi_sample, pi_act):
                return pi.log_p(pi_sample) - tf.reduce_sum(tf.log(1-tf.square(pi_act)+1e-6), axis=-1, keepdims=True)
            with tf.variable_scope('policy'):
                self.pi = mlp_policy(self.obs_ph)
                self.pi_sample = self.pi.mean+self.pi_noise_ph*self.pi.std
                self.pi_act = tf.tanh(self.pi_sample)
                self.pi_log_p = get_pi_log_p(self.pi, self.pi_sample, self.pi_act)
            with tf.variable_scope('policy', reuse=True):
                self.pi_next = mlp_policy(self.obs_next_ph)
                self.pi_next_sample = self.pi_next.mean+self.pi_next_noise_ph*self.pi_next.std
                self.pi_next_act = tf.tanh(self.pi_next_sample)
                self.pi_next_log_p = get_pi_log_p(self.pi_next, self.pi_next_sample, self.pi_next_act)

            with tf.variable_scope('q_value_1'):
                self.q_1 = mlp_q_value(self.obs_ph, self.acts_ph)
            with tf.variable_scope('q_value_2'):
                self.q_2 = mlp_q_value(self.obs_ph, self.acts_ph)
            with tf.variable_scope('q_value_1', reuse=True):
                self.q_pi_1 = mlp_q_value(self.obs_ph, self.pi_act)
            with tf.variable_scope('q_value_2', reuse=True):
                self.q_pi_2 = mlp_q_value(self.obs_ph, self.pi_act)
            self.q_pi = tf.minimum(self.q_pi_1, self.q_pi_2) - self.alpha*self.pi_log_p

        with tf.variable_scope('target'):
            with tf.variable_scope('q_value_1'):
                self.q_t_1 = mlp_q_value(self.obs_next_ph, self.pi_next_act)
            with tf.variable_scope('q_value_2'):
                self.q_t_2 = mlp_q_value(self.obs_next_ph, self.pi_next_act)
            self.q_t = tf.minimum(self.q_t_1, self.q_t_2) - self.alpha*self.pi_next_log_p

    def create_operators(self):
        self.pi_loss = tf.reduce_mean(-self.q_pi)
        self.pi_optimizer = tf.train.AdamOptimizer(self.args.pi_lr)
        self.pi_train_op = self.pi_optimizer.minimize(self.pi_loss, var_list=get_vars('main/policy'))

        if self.args.alpha_lr>0:
            self.alpha_loss = -self.log_alpha*tf.stop_gradient(tf.reduce_mean(self.pi_log_p) - np.prod(self.args.acts_dims))
            self.alpha_optimizer = tf.train.AdamOptimizer(self.args.alpha_lr)
            self.alpha_train_op  = self.alpha_optimizer.minimize(self.alpha_loss, var_list=get_vars('main/alpha'))
            self.pi_train_op = tf.group([self.pi_train_op, self.alpha_train_op])

        q_target = tf.stop_gradient(self.rews_ph+(1.0-self.done_ph)*self.args.gamma*self.q_t)
        self.q_loss = tf.reduce_mean(tf.square(self.q_1-q_target)) + tf.reduce_mean(tf.square(self.q_2-q_target))
        self.q_optimizer = tf.train.AdamOptimizer(self.args.q_lr)
        self.q_train_op = self.q_optimizer.minimize(self.q_loss, var_list=get_vars('main/q_value'))

        self.target_update_op = tf.group([
            q_t.assign(self.args.polyak*q_t + (1.0-self.args.polyak)*q)
            for q, q_t in zip(get_vars('main/q_value'), get_vars('target/q_value'))
        ])

        self.saver=tf.train.Saver()
        self.init_op = tf.global_variables_initializer()
        self.target_init_op = tf.group([
            q_t.assign(q)
            for q, q_t in zip(get_vars('main/q_value'), get_vars('target/q_value'))
        ])

    def step(self, obs, explore=False, test_info=False):
        if explore:
            noise = np.random.normal(0.0, 1.0, size=self.args.acts_dims)
        else:
            noise = np.zeros(shape=self.args.acts_dims, dtype=np.float32)
        feed_dict = {
            self.raw_obs_ph: [obs],
            self.pi_noise_ph: [noise]
        }
        action, info = self.sess.run([self.pi_act, self.step_info], feed_dict)
        action = action[0]

        if test_info: return action, info
        return action

    def feed_dict(self, batch):
        noise_size = np.array(batch['acts']).shape
        return {
            **super().feed_dict(batch),
            self.pi_noise_ph: np.random.normal(0.0, 1.0, size=noise_size),
            self.pi_next_noise_ph: np.random.normal(0.0, 1.0, size=noise_size)
        }
