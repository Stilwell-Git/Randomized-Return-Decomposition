import numpy as np
import tensorflow as tf
from utils.tf_utils import get_vars
from .base import Base

class DQN(Base):
    def __init__(self, args):
        self.args = args
        self.acts_num = args.acts_dims[0]
        self.create_model()

        self.train_info_q = {
            'Q_loss': self.q_loss
        }
        self.train_info = { **self.train_info_q }
        self.step_info = {
            'Q_average': self.q_pi
        }

    def create_network(self):
        def mlp_value(obs_ph):
            with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                q_dense1 = tf.layers.dense(obs_ph, 256, activation=tf.nn.relu, name='q_dense1')
                q_dense2 = tf.layers.dense(q_dense1, 256, activation=tf.nn.relu, name='q_dense2')
                q = tf.layers.dense(q_dense2, self.acts_num, name='q')
            return q

        def conv_value(obs_ph):
            with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                q_conv1 = tf.layers.conv2d(obs_ph, 32, 8, 4, 'same', activation=tf.nn.relu, name='q_conv1')
                q_conv2 = tf.layers.conv2d(q_conv1, 64, 4, 2, 'same', activation=tf.nn.relu, name='q_conv2')
                q_conv3 = tf.layers.conv2d(q_conv2, 64, 3, 1, 'same', activation=tf.nn.relu, name='q_conv3')
                q_conv3_flat = tf.layers.flatten(q_conv3)

                q_dense_act = tf.layers.dense(q_conv3_flat, 512, activation=tf.nn.relu, name='q_dense_act')
                q_act = tf.layers.dense(q_dense_act, self.acts_num, name='q_act')

                if self.args.dueling:
                    q_dense_base = tf.layers.dense(q_conv3_flat, 512, activation=tf.nn.relu, name='q_dense_base')
                    q_base = tf.layers.dense(q_dense_base, 1, name='q_base')
                    q = q_base + q_act - tf.reduce_mean(q_act, axis=1, keepdims=True)
                else:
                    q = q_act
            return q

        value_net = mlp_value if len(self.args.obs_dims)==1 else conv_value

        with tf.variable_scope('main'):
            with tf.variable_scope('value'):
                self.q = value_net(self.obs_ph)
                self.q_pi = tf.reduce_max(self.q, axis=1, keepdims=True)
            if self.args.double:
                with tf.variable_scope('value', reuse=True):
                    self.q_next = value_net(self.obs_next_ph)
                    self.pi_next = tf.one_hot(tf.argmax(self.q_next, axis=1), self.acts_num, dtype=tf.float32)

        with tf.variable_scope('target'):
            with tf.variable_scope('value'):
                if self.args.double:
                    self.q_t = tf.reduce_sum(value_net(self.obs_next_ph)*self.pi_next, axis=1, keepdims=True)
                else:
                    self.q_t = tf.reduce_max(value_net(self.obs_next_ph), axis=1, keepdims=True)

    def create_operators(self):
        target = tf.stop_gradient(self.rews_ph+(1.0-self.done_ph)*self.args.gamma*self.q_t)
        self.q_acts = tf.reduce_sum(self.q*self.acts_ph, axis=1, keepdims=True)
        self.q_loss = tf.losses.huber_loss(target, self.q_acts)
        if self.args.optimizer=='adam':
            self.q_optimizer = tf.train.AdamOptimizer(self.args.q_lr, epsilon=self.args.Adam_eps)
        elif self.args.optimizer=='rmsprop':
            self.q_optimizer = tf.train.RMSPropOptimizer(self.args.q_lr, decay=self.args.RMSProp_decay, epsilon=self.args.RMSProp_eps)
        self.q_train_op = self.q_optimizer.minimize(self.q_loss, var_list=get_vars('main/value'))

        self.target_update_op = tf.group([
            v_t.assign(v)
            for v, v_t in zip(get_vars('main'), get_vars('target'))
        ])

        self.saver=tf.train.Saver()
        self.init_op = tf.global_variables_initializer()
        self.target_init_op = tf.group([
            v_t.assign(v)
            for v, v_t in zip(get_vars('main'), get_vars('target'))
        ])

    def step(self, obs, explore=False, test_info=False):
        if (not test_info) and (self.args.buffer.step_counter<self.args.warmup):
            return np.random.randint(self.acts_num)

        # eps-greedy exploration
        if explore and np.random.uniform()<=self.args.eps_act:
            return np.random.randint(self.acts_num)

        feed_dict = {
            # the same processing as frame_stack_buffer
            self.raw_obs_ph: [obs/255.0]
        }
        q_value, info = self.sess.run([self.q, self.step_info], feed_dict)
        action = np.argmax(q_value[0])

        if test_info: return action, info
        return action

    def feed_dict(self, batch):
        def one_hot(idx):
            idx = np.array(idx)
            batch_size = idx.shape[0]
            res = np.zeros((batch_size, self.acts_num), dtype=np.float32)
            res[np.arange(batch_size),idx] = 1.0
            return res
        feed_dict = {
            self.raw_obs_ph: np.array(batch['obs']),
            self.raw_obs_next_ph: np.array(batch['obs_next']),
            self.acts_ph: one_hot(batch['acts']),
            self.rews_ph: np.array(batch['rews']),
            self.done_ph: batch['done']
        }
        return feed_dict

    def train(self, batch):
        feed_dict = self.feed_dict(batch)
        info, _ = self.sess.run([self.train_info, self.q_train_op], feed_dict)
        return info
