import numpy as np
import tensorflow as tf
from utils.tf_utils import get_vars, get_reg_loss
from .base import Base

class DDPG(Base):
    def __init__(self, args):
        self.args = args
        self.create_model()

        self.train_info_pi = {
            'Pi_loss': self.pi_loss
        }
        self.train_info_q = {
            'Q_loss': self.q_loss,
            'Q_reg_loss': self.q_reg_loss
        }
        self.train_info = {**self.train_info_pi, **self.train_info_q}

        self.step_info = {
            'Q_average': self.q_pi
        }

    def create_network(self):
        def mlp_policy(obs_ph):
            with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                pi_dense1 = tf.layers.dense(obs_ph, 400, activation=tf.nn.relu, name='pi_dense1')
                pi_dense2 = tf.layers.dense(pi_dense1, 300, activation=tf.nn.relu, name='pi_dense2')
                pi = tf.layers.dense(pi_dense2, self.args.acts_dims[0], activation=tf.nn.tanh, name='pi')
            return pi

        def mlp_value(obs_ph, acts_ph):
            with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                q_dense1 = tf.layers.dense(obs_ph, 400, activation=tf.nn.relu, name='q_dense1')
                q_dense2 = tf.layers.dense(tf.concat([q_dense1, acts_ph], axis=1), 300, activation=tf.nn.relu, name='q_dense2')
                q = tf.layers.dense(q_dense2, 1, name='q')
            return q

        with tf.variable_scope('main'):
            with tf.variable_scope('policy'):
                self.pi = mlp_policy(self.obs_ph)
            with tf.variable_scope('value', regularizer=tf.contrib.layers.l2_regularizer(self.args.q_reg)):
                self.q = mlp_value(self.obs_ph, self.acts_ph)
            with tf.variable_scope('value', reuse=True):
                self.q_pi = mlp_value(self.obs_ph, self.pi)

        with tf.variable_scope('target'):
            with tf.variable_scope('policy'):
                self.pi_t = mlp_policy(self.obs_next_ph)
            with tf.variable_scope('value'):
                self.q_t = mlp_value(self.obs_next_ph, self.pi_t)

    def create_operators(self):
        self.pi_loss = -tf.reduce_mean(self.q_pi)
        self.pi_optimizer = tf.train.AdamOptimizer(self.args.pi_lr)
        self.pi_train_op = self.pi_optimizer.minimize(self.pi_loss, var_list=get_vars('main/policy'))

        target = tf.stop_gradient(self.rews_ph+(1.0-self.done_ph)*self.args.gamma*self.q_t)
        self.q_loss = tf.reduce_mean(tf.square(self.q-target))
        self.q_reg_loss = get_reg_loss('main/value')
        self.q_optimizer = tf.train.AdamOptimizer(self.args.q_lr)
        self.q_train_op = self.q_optimizer.minimize(self.q_loss + self.q_reg_loss, var_list=get_vars('main/value'))

        self.target_update_op = tf.group([
            v_t.assign(self.args.polyak*v_t + (1.0-self.args.polyak)*v)
            for v, v_t in zip(get_vars('main'), get_vars('target'))
        ])

        self.saver = tf.train.Saver()
        self.init_op = tf.global_variables_initializer()
        self.target_init_op = tf.group([
            v_t.assign(v)
            for v, v_t in zip(get_vars('main'), get_vars('target'))
        ])

    def step(self, obs, explore=False, test_info=False):
        feed_dict = {
            self.raw_obs_ph: [obs]
        }
        action, info = self.sess.run([self.pi, self.step_info], feed_dict)
        action = action[0]

        # uncorrelated gaussian explorarion
        if explore: action += np.random.normal(0, self.args.std_act, size=self.args.acts_dims)
        action = np.clip(action, -1, 1)

        if test_info: return action, info
        return action

    def feed_dict(self, batch):
        return {
            self.raw_obs_ph: batch['obs'],
            self.raw_obs_next_ph: batch['obs_next'],
            self.acts_ph: batch['acts'],
            self.rews_ph: batch['rews'],
            self.done_ph: batch['done']
        }

    def train(self, batch):
        feed_dict = self.feed_dict(batch)
        info, _, _ = self.sess.run([self.train_info, self.pi_train_op, self.q_train_op], feed_dict)
        return info

    def train_pi(self, batch):
        feed_dict = self.feed_dict(batch)
        info, _ = self.sess.run([self.train_info_pi, self.pi_train_op], feed_dict)
        return info

    def train_q(self, batch):
        feed_dict = self.feed_dict(batch)
        info, _ = self.sess.run([self.train_info_q, self.q_train_op], feed_dict)
        return info
