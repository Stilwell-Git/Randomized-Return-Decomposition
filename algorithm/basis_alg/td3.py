import numpy as np
import tensorflow as tf
from utils.tf_utils import get_vars
from .ddpg import DDPG

class TD3(DDPG):
    def __init__(self, args):
        self.args = args
        self.create_model()

        self.train_info_pi = {
            'Pi_loss': self.pi_loss
        }
        self.train_info_q = {
            'Q_loss': self.q_loss,
        }
        self.train_info = {**self.train_info_pi, **self.train_info_q}

        self.step_info = {
            'Q_average': self.q_pi
        }

    def create_inputs(self):
        super().create_inputs()

        self.target_noise_ph = tf.placeholder(tf.float32, [None]+self.args.acts_dims)

    def create_network(self):
        def mlp_policy(obs_ph):
            with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                pi_dense1 = tf.layers.dense(obs_ph, 256, activation=tf.nn.relu, name='pi_dense1')
                pi_dense2 = tf.layers.dense(pi_dense1, 256, activation=tf.nn.relu, name='pi_dense2')
                pi = tf.layers.dense(pi_dense2, self.args.acts_dims[0], activation=tf.nn.tanh, name='pi')
            return pi

        def mlp_value(obs_ph, acts_ph):
            state_ph = tf.concat([obs_ph, acts_ph], axis=1)
            with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                q_dense1 = tf.layers.dense(state_ph, 256, activation=tf.nn.relu, name='q_dense1')
                q_dense2 = tf.layers.dense(q_dense1, 256, activation=tf.nn.relu, name='q_dense2')
                q = tf.layers.dense(q_dense2, 1, name='q')
            return q

        with tf.variable_scope('main'):
            with tf.variable_scope('policy'):
                self.pi = mlp_policy(self.obs_ph)
            with tf.variable_scope('value_1'):
                self.q_1 = mlp_value(self.obs_ph, self.acts_ph)
            with tf.variable_scope('value_2'):
                self.q_2 = mlp_value(self.obs_ph, self.acts_ph)
            with tf.variable_scope('value_1', reuse=True):
                self.q_pi = mlp_value(self.obs_ph, self.pi)
            with tf.variable_scope('value_2', reuse=True):
                self.q_pi_2 = mlp_value(self.obs_ph, self.pi)

        with tf.variable_scope('target'):
            with tf.variable_scope('policy'):
                self.pi_t = tf.clip_by_value(mlp_policy(self.obs_next_ph)+self.target_noise_ph, -1.0, 1.0)
            with tf.variable_scope('value_1'):
                self.q_t_1 = mlp_value(self.obs_next_ph, self.pi_t)
            with tf.variable_scope('value_2'):
                self.q_t_2 = mlp_value(self.obs_next_ph, self.pi_t)
            self.q_t = tf.minimum(self.q_t_1, self.q_t_2)

    def create_operators(self):
        self.pi_loss = -tf.reduce_mean(self.q_pi)
        self.pi_optimizer = tf.train.AdamOptimizer(self.args.pi_lr)
        self.pi_train_op = self.pi_optimizer.minimize(self.pi_loss, var_list=get_vars('main/policy'))

        target = tf.stop_gradient(self.rews_ph+(1.0-self.done_ph)*self.args.gamma*self.q_t)
        self.q_loss = tf.reduce_mean(tf.square(self.q_1-target)) + tf.reduce_mean(tf.square(self.q_2-target))
        self.q_optimizer = tf.train.AdamOptimizer(self.args.q_lr)
        self.q_train_op = self.q_optimizer.minimize(self.q_loss, var_list=get_vars('main/value'))

        self.target_update_op = tf.group([
            v_t.assign(self.args.polyak*v_t + (1.0-self.args.polyak)*v)
            for v, v_t in zip(get_vars('main/'), get_vars('target/'))
        ])

        self.saver = tf.train.Saver()
        self.init_op = tf.global_variables_initializer()
        self.target_init_op = tf.group([
            v_t.assign(v)
            for v, v_t in zip(get_vars('main/'), get_vars('target/'))
        ])

    def feed_dict(self, batch):
        noise_size = np.array(batch['acts']).shape
        return {
            **super().feed_dict(batch),
            self.target_noise_ph: np.clip(np.random.normal(0.0, self.args.pi_t_std, size=noise_size), -self.args.pi_t_clip, self.args.pi_t_clip)
        }
