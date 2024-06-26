import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

class WGAN:
    def __init__(self, data, seq_len=96, feature_dim=1, z_dim=100, batch_size=64, learning_rate=1e-4, gradient_penalty_weight=10.0):
        self.data = self.normalize_data(data)
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gradient_penalty_weight = gradient_penalty_weight

        self.build_model()

    def normalize_data(self, data):
        self.data_min = data.min()
        self.data_max = data.max()
        return (data - self.data_min) / (self.data_max - self.data_min) * 2 - 1

    def denormalize_data(self, data):
        return (data + 1) / 2 * (self.data_max - self.data_min) + self.data_min

    def build_generator(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.z_dim,)))
        model.add(tf.keras.layers.Dense(self.seq_len * self.feature_dim, activation='relu', kernel_initializer='he_normal'))
        model.add(tf.keras.layers.Reshape((self.seq_len, self.feature_dim)))
        model.add(tf.keras.layers.Conv1D(64, kernel_size=5, strides=1, padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(128, kernel_size=5, strides=1, padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(self.feature_dim, kernel_size=5, strides=1, padding='same', activation='tanh'))
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.seq_len, self.feature_dim)))
        model.add(tf.keras.layers.Conv1D(64, kernel_size=5, strides=2, padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(128, kernel_size=5, strides=2, padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation=None))
        return model

    def build_model(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.real_data = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.feature_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        self.fake_data = self.generator(self.z)
        self.real_logits = self.discriminator(self.real_data)
        self.fake_logits = self.discriminator(self.fake_data)

        self.d_loss_real = tf.reduce_mean(self.real_logits)
        self.d_loss_fake = tf.reduce_mean(self.fake_logits)
        self.d_loss = self.d_loss_fake - self.d_loss_real

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=1.)
        interpolates = alpha * self.real_data + ((1 - alpha) * self.fake_data)
        d_interpolates_logits = self.discriminator(interpolates)
        gradients = tf.gradients(d_interpolates_logits, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.d_loss += self.gradient_penalty_weight * gradient_penalty

        self.g_loss = -self.d_loss_fake

        d_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9)

        d_gradients = d_optimizer.compute_gradients(self.d_loss, var_list=self.discriminator.trainable_variables)
        g_gradients = g_optimizer.compute_gradients(self.g_loss, var_list=self.generator.trainable_variables)

        d_clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in d_gradients]
        g_clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in g_gradients]

        self.d_train_op = d_optimizer.apply_gradients(d_clipped_gradients)
        self.g_train_op = g_optimizer.apply_gradients(g_clipped_gradients)

    def sample_z(self, batch_size):
        return np.random.uniform(-1., 1., size=[batch_size, self.z_dim])

    def sample_images(self, epoch, sample_dir, sess):
        z_sample = self.sample_z(16)
        gen_samples = sess.run(self.fake_data, feed_dict={self.z: z_sample})
        gen_samples = self.denormalize_data(gen_samples)

        for i, sample in enumerate(gen_samples):
            plt.plot(sample)
            plt.savefig(f'{sample_dir}/sample_{epoch}_{i}.png')
            plt.close()

    def train(self, epochs, sample_dir, save_model_path):
        os.makedirs(sample_dir, exist_ok=True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                for _ in range(5):
                    batch_idx = np.random.randint(0, self.data.shape[0], size=self.batch_size)
                    real_batch = self.data[batch_idx]
                    z_batch = self.sample_z(self.batch_size)

                    _, d_loss_curr = sess.run([self.d_train_op, self.d_loss], feed_dict={self.real_data: real_batch, self.z: z_batch})

                z_batch = self.sample_z(self.batch_size)
                _, g_loss_curr = sess.run([self.g_train_op, self.g_loss], feed_dict={self.z: z_batch})

                if epoch % 100 == 0 or epoch < 100:
                    print(f'Epoch: {epoch}, D loss: {d_loss_curr}, G loss: {g_loss_curr}')

                if epoch % 1000 == 0:
                    self.sample_images(epoch, sample_dir, sess)

            saver = tf.train.Saver()
            saver.save(sess, save_model_path)

if __name__ == '__main__':
    data = np.load('ca_data_99solar_15min.npy')
    data = data.reshape(-1, 96, 1)  # Reshape data for time series format

    sample_dir = 'Samples/wgan_timeseries'
    save_model_path = 'Models/wgan_timeseries.ckpt'

    wgan = WGAN(data, seq_len=96, feature_dim=1, z_dim=100, batch_size=128, learning_rate=1e-4)
    wgan.train(epochs=30000, sample_dir=sample_dir, save_model_path=save_model_path)
