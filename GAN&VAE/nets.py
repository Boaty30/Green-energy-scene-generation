import tensorflow as tf
import tensorflow.contrib.layers as tcl

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class G_rnn_timeseries(object):
    def __init__(self):
        self.name = 'G_rnn_timeseries'

    def __call__(self, z, seq_len):
        with tf.variable_scope(self.name) as scope:
            cell = tf.nn.rnn_cell.LSTMCell(num_units=128, activation=tf.nn.relu)
            z = tf.expand_dims(z, axis=-1)
            outputs, _ = tf.nn.dynamic_rnn(cell, z, dtype=tf.float32)
            outputs = tcl.fully_connected(outputs, num_outputs=1, activation_fn=None)
            return outputs

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class D_rnn_timeseries(object):
    def __init__(self):
        self.name = 'D_rnn_timeseries'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            cell = tf.nn.rnn_cell.LSTMCell(num_units=128, activation=leaky_relu)
            outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            outputs = tf.layers.flatten(outputs)
            logits = tcl.fully_connected(outputs, num_outputs=1, activation_fn=None)
            return logits, None

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
