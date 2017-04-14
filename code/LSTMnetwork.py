import tensorflow as tf
import numpy as np

class LSTMNetwork:

    def __init__(self):
        self.X = tf.placeholder(tf.float32, [])