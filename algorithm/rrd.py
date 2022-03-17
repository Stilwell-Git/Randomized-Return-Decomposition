import numpy as np
import tensorflow as tf
from utils.tf_utils import get_vars
from algorithm import basis_algorithm_collection

def RRD(args):
    basis_alg_class = basis_algorithm_collection[args.basis_alg]
    class RandomizedReturnDecomposition(basis_alg_class):
        def __init__(self, args):
            super().__init__(args)

            self.train_info_r = {
                'R_loss': self.r_loss
            }
            if args.rrd_bias_correction:
                self.train_info_r['R_var'] = self.r_var
            self.train_info_q = {**self.train_info_q, **self.train_info_r}
            self.train_info = {**self.train_info, **self.train_info_r}

        def create_inputs(self):
            super().create_inputs()

            self.rrd_raw_obs_ph = tf.placeholder(tf.float32, [None, self.args.rrd_sample_size]+self.args.obs_dims)
            self.rrd_raw_obs_next_ph = tf.placeholder(tf.float32, [None, self.args.rrd_sample_size]+self.args.obs_dims)
            self.rrd_acts_ph = tf.placeholder(tf.float32, [None, self.args.rrd_sample_size]+self.args.acts_dims)
            self.rrd_rews_ph = tf.placeholder(tf.float32, [None, 1])
            if self.args.rrd_bias_correction:
                self.rrd_var_coef_ph = tf.placeholder(tf.float32, [None, 1])

        def create_normalizer(self):
            super().create_normalizer()

            if self.args.obs_normalization:
                self.rrd_obs_ph = self.obs_normalizer.normalize(self.rrd_raw_obs_ph)
                self.rrd_obs_next_ph = self.obs_normalizer.normalize(self.rrd_raw_obs_next_ph)
            else:
                self.rrd_obs_ph = self.rrd_raw_obs_ph
                self.rrd_obs_next_ph = self.rrd_raw_obs_next_ph

        def create_network(self):
            super().create_network()

            def mlp_rrd(rrd_obs_ph, rrd_acts_ph, rrd_obs_next_ph):
                rrd_state_ph = tf.concat([rrd_obs_ph, rrd_acts_ph, rrd_obs_ph-rrd_obs_next_ph], axis=-1)
                with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                    r_dense1 = tf.layers.dense(rrd_state_ph, 256, activation=tf.nn.relu, name='r_dense1')
                    r_dense2 = tf.layers.dense(r_dense1, 256, activation=tf.nn.relu, name='r_dense2')
                    r = tf.layers.dense(r_dense2, 1, name='r')
                return r

            def conv_rrd(rrd_obs_ph, rrd_acts_ph, rrd_obs_next_ph):
                flatten = (len(list(rrd_obs_ph.shape))==len(self.args.obs_dims)+2)
                rrd_state_ph = tf.concat([rrd_obs_ph, rrd_obs_ph-rrd_obs_next_ph], axis=-1)
                if flatten:
                    rrd_state_ph = tf.reshape(rrd_state_ph, [-1]+self.args.obs_dims[:-1]+[self.args.obs_dims[-1]*2])
                    rrd_acts_ph = tf.reshape(rrd_acts_ph, [-1]+self.args.acts_dims)
                with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                    r_conv1 = tf.layers.conv2d(rrd_state_ph, 32, 8, 4, 'same', activation=tf.nn.relu, name='r_conv1')
                    r_conv2 = tf.layers.conv2d(r_conv1, 64, 4, 2, 'same', activation=tf.nn.relu, name='r_conv2')
                    r_conv3 = tf.layers.conv2d(r_conv2, 64, 3, 1, 'same', activation=tf.nn.relu, name='r_conv3')
                    r_conv3_flat = tf.layers.flatten(r_conv3)

                    r_dense_act = tf.layers.dense(r_conv3_flat, 512, activation=tf.nn.relu, name='r_dense_act')
                    r_act = tf.layers.dense(r_dense_act, self.acts_num, name='r_act')
                    r = tf.reduce_sum(r_act*rrd_acts_ph, axis=-1, keepdims=True)
                if flatten:
                    r = tf.reshape(r, [-1, self.args.rrd_sample_size, 1])
                return r

            reward_net = mlp_rrd if len(self.args.obs_dims)==1 else conv_rrd
            with tf.variable_scope('rrd'):
                self.rrd_rews_pred = reward_net(self.rrd_obs_ph, self.rrd_acts_ph, self.rrd_obs_next_ph)
                self.rrd = tf.reduce_mean(self.rrd_rews_pred, axis=1)
            with tf.variable_scope('rrd', reuse=True):
                self.rews_ph = self.rews_pred = tf.stop_gradient(reward_net(self.obs_ph, self.acts_ph, self.obs_next_ph))

        def create_operators(self):
            super().create_operators()

            self.r_loss = tf.reduce_mean(tf.square(self.rrd-self.rrd_rews_ph))
            if self.args.rrd_bias_correction:
                assert self.args.rrd_sample_size>1
                n = self.args.rrd_sample_size
                self.r_var_single = tf.reduce_sum(tf.square(self.rrd_rews_pred-tf.reduce_mean(self.rrd_rews_pred, axis=1, keepdims=True)), axis=1) / (n-1)
                self.r_var = tf.reduce_mean(self.r_var_single*self.rrd_var_coef_ph/n)
                self.r_total_loss = self.r_loss - self.r_var
            else:
                self.r_total_loss = self.r_loss
            self.r_optimizer = tf.train.AdamOptimizer(self.args.r_lr)
            self.r_train_op = self.r_optimizer.minimize(self.r_total_loss, var_list=get_vars('rrd/'))
            self.q_train_op = tf.group([self.q_train_op, self.r_train_op])

            self.init_op = tf.global_variables_initializer()

        def feed_dict(self, batch):
            batch_size = np.array(batch['obs']).shape[0]
            basis_feed_dict = super().feed_dict(batch)
            del basis_feed_dict[self.rews_ph]
            def one_hot(idx):
                idx = np.array(idx)
                batch_size, sample_size = idx.shape[0], idx.shape[1]
                idx = np.reshape(idx, [batch_size*sample_size])
                res = np.zeros((batch_size*sample_size, self.acts_num), dtype=np.float32)
                res[np.arange(batch_size*sample_size),idx] = 1.0
                res = np.reshape(res, [batch_size, sample_size, self.acts_num])
                return res
            rrd_feed_dict = {
                **basis_feed_dict, **{
                    self.rrd_raw_obs_ph: batch['rrd_obs'],
                    self.rrd_raw_obs_next_ph: batch['rrd_obs_next'],
                    self.rrd_acts_ph: batch['rrd_acts'] if self.args.env_category!='atari' else one_hot(batch['rrd_acts']),
                    self.rrd_rews_ph: batch['rrd_rews'],
                }
            }
            if self.args.rrd_bias_correction:
                rrd_feed_dict[self.rrd_var_coef_ph] = batch['rrd_var_coef']
            return rrd_feed_dict

    return RandomizedReturnDecomposition(args)
