# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import layers

sensor_image_size = (50, 50)


class CNNNetwork:

    def __init__(self):

        # Build Model
        # [batch_size, (2d_image_size), num_of_features]
        self.X = tf.placeholder(tf.float32, [None, sensor_image_size[0], sensor_image_size[1], 1])
        # [batch_size, num_of_outputs]
        self.Y = tf.placeholder(tf.float32, [None, 2])

        self.L1_filters = 32
        self.L1_kernel = [3, 3]
        self.L1 = layers.conv2d(self.X, self.L1_filters, self.L1_kernel)

        self.L2_pool = [2, 2]
        self.L2 = layers.max_pool2d(self.L1, self.L2_pool)

        self.L3_filters = 64
        self.L3_kernel = [3, 3]
        self.L3 = layers.conv2d(self.L2, self.L3_filters, self.L3_kernel,
                                normalizer_fn=tf.nn.dropout, normalizer_params={'keep_prob': 0.8})

        self.L4_pool = [2, 2]
        self.L4 = layers.max_pool2d(self.L3, self.L4_pool)

        self.L5 = layers.flatten(self.L4)
        self.L5 = layers.fully_connected(self.L5, self.L5.size, normalizer_fn=layers.batch_norm)

        self.model = layers.fully_connected(self.L5, 2)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    def init_session(self):
        pass

    def save_model(self):
        pass

    def train(self, x_batch, y_batch):
        _, loss = self.session.run([self.optimizer, self.cose], feed_dict={self.X: x_batch, self.Y: y_batch})
        print('cost: ', '{:.6f}'.format(loss))

    def predict(self, x, y):
        prediction = tf.cast(tf.argmax(self.logits, 1), tf.int32)
        prediction_check = tf.equal(prediction, self.labels)
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

        real, predict, accuracy_val = self.session.run([self.labels, prediction_check, accuracy], feed_dict={self.X:x, self.Y:y})
        return real, predict, accuracy_val

