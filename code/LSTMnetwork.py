# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn
from LSTMSettings import *
import numpy as np


class LSTMNetwork:

    def __init__(self):

        # Build model
        # [batch_size, time_steps, input_size]
        self.X = tf.placeholder(tf.float32, [None, None, n_sensors])
        # [batch_size, time_steps]
        self.Y = tf.placeholder(tf.int32, [None, 1])
        # [time_steps, batch_size, input_size]
        self.X_t = tf.transpose(self.X, [1, 0, 2])

        # LSTM cell
        #self.cell = rnn.BasicLSTMCell(n_hidden)
        #self.cell = rnn.DropoutWrapper(self.cell, output_keep_prob=0.5)
        self.cell = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden), output_keep_prob=0.5) for _ in range(n_layers)])
        self.outputs, _ = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.X_t,
                                            dtype=tf.float32, time_major=True)

        self.W = tf.Variable(tf.random_normal([n_hidden, n_sensor_class]))
        self.b = tf.Variable(tf.random_normal([n_sensor_class]))

        # 마지막 output을 기준으로 양/불을 판단
        self.logits = tf.matmul(self.outputs[-1], self.W) + self.b
        self.labels = tf.reshape(self.Y, [-1])

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)

        #######################

        # TODO : LEARN HOW THESE WORK
        self.time_step = 0
        self.saver, self.session = self.init_session()
        # self.writer = tf.summary.FileWriter('LSTMlogs', self.session.graph)
        # self.summary = tf.summary.merge_all()

    def init_session(self, restore_path=''):
        saver = tf.train.Saver()
        session = tf.Session()

        session.run(tf.global_variables_initializer())
        print("Initializing new model...")

        # TODO : IMPLEMENT THIS
        # try:
        #     saver.restore(session, restore_path)
        #     print("Restoring model...")
        # except:
        #     session.run(tf.global_variables_initializer())
        #     print("Initializing new model...")

        return saver, session

    def save_model(self):
        save_path = self.saver.save(self.session, "LSTMmodel")
        print("Model saved in file: %s" % save_path)

    def train(self, x_batch, y_batch):

        _, loss = self.session.run([self.optimizer, self.cost], feed_dict={self.X: x_batch, self.Y: y_batch})
        print('cost: ', '{:.6f}'.format(loss))

    def predict(self, x):
        prediction = tf.cast(tf.argmax(self.logits, 1), tf.int32)

        predict = self.session.run([prediction], feed_dict={self.X: x , self.Y: y})
        return predict

    def test(self, x, y):
        prediction = tf.cast(tf.argmax(self.logits, 1), tf.int32)
        prediction_check = tf.equal(prediction, self.labels)
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

        real, predict, accuracy_val = self.session.run([self.labels, prediction, accuracy], feed_dict={self.X: x , self.Y: y})
        return real, predict, accuracy_val
