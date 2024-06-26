import numpy as np
import tensorflow as tf

class TimeseriesData:
    def __init__(self, filepath):
        self.data = np.load(filepath)
        self.data = self.data.reshape(-1, 96, 1)
        
        self.dataset = tf.data.Dataset.from_tensor_slices(self.data)
        self.dataset = self.dataset.shuffle(buffer_size=10000).repeat().batch(512)
        self.iterator = self.dataset.make_one_shot_iterator()

    def __call__(self):
        return self.iterator.get_next()
