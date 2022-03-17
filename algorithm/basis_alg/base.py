import numpy as np
import tensorflow as tf
from utils.tf_utils import Normalizer

class Base:
    def __init__(self, args):
        self.args = args
        self.create_model()

    def create_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def create_inputs(self):
        self.raw_obs_ph = tf.placeholder(tf.float32, [None]+self.args.obs_dims)
        self.raw_obs_next_ph = tf.placeholder(tf.float32, [None]+self.args.obs_dims)
        self.acts_ph = tf.placeholder(tf.float32, [None]+self.args.acts_dims)
        self.rews_ph = tf.placeholder(tf.float32, [None, 1])
        self.done_ph = tf.placeholder(tf.float32, [None, 1])

    def create_normalizer(self):
        if self.args.obs_normalization:
            with tf.variable_scope('normalizer'):
                self.obs_normalizer = Normalizer(self.args.obs_dims, self.sess)
            self.obs_ph = self.obs_normalizer.normalize(self.raw_obs_ph)
            self.obs_next_ph = self.obs_normalizer.normalize(self.raw_obs_next_ph)
        else:
            self.obs_normalizer = None
            self.obs_ph = self.raw_obs_ph
            self.obs_next_ph = self.raw_obs_next_ph

    def create_network(self):
        raise NotImplementedError

    def create_operators(self):
        raise NotImplementedError

    def create_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_session()
            self.create_inputs()
            self.create_normalizer()
            self.create_network()
            self.create_operators()
        self.init_network()

    def init_network(self):
        self.sess.run(self.init_op)
        self.sess.run(self.target_init_op)

    def normalizer_update(self, batch):
        if self.args.obs_normalization:
            self.obs_normalizer.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))

    def target_update(self):
        self.sess.run(self.target_update_op)

    def save_model(self, save_path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, save_path)

    def load_model(self, load_path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, load_path)
