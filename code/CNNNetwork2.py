# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import layers

batch_size = 10
image_size = 28

# similar to LeNet5
class LeNet5Network:

    def __init__(self):
        self.X = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 1])

        L1_kernel = 5
        L1_filter = 16
        self.W1 = tf.Variable(tf.random_normal([L1_kernel, L1_kernel, 1, L1_filter], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([L1_filter]))

        self.L1_conv = tf.nn.conv2d(self.X, self.W1, strides=[1, 1, 1, 1], padding='SAME')
        self.L1_conv = tf.nn.relu(self.L1_conv + self.b1)
        self.L1_conv = tf.nn.dropout(self.L1_conv, keep_prob=0.8)
        self.L1_pool = tf.nn.max_pool(self.L1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.L1_norm = tf.nn.lrn(self.L1.pool, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)

        L2_kernel = 3
        L2_filter = 16

        pass