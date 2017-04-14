import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

n_hidden = 100  # magic number
n_layers = 5  # magic number
n_sensors = 83  # num of input types
n_sensor_class = 2  # num of output types

# Network model
# [batch_size, time_steps, input_size]
# [batch_size, time_steps]
X = tf.placeholder(tf.float32, [None, None, n_sensors])
Y = tf.placeholder(tf.float32, [None, 1])
# [time_steps, batch_size, input_size]
X_t = tf.transpose(X, [1, 0, 2])

W = tf.Variable(tf.random_normal([n_hidden, n_sensor_class]))
b = tf.Variable(tf.random_normal([n_sensor_class]))

# LSTM cell
cell = rnn.BasicLSTMCell(n_hidden)
cell = rnn.DropoutWrapper(cell, output_keep_prob=0.5)
cell = rnn.MultiRNNCell([cell] * n_layers)
outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=X_t, dtype=tf.float32, time_major=True)

logits = tf.matmul(outputs[-1], W) + b
labels = tf.reshape(Y, [-1])

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())