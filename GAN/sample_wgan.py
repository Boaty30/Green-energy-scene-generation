import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from wgan_timeseries_new import WGAN

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def plot_samples(gen_samples, real_data, sample_dir):
    os.makedirs(sample_dir, exist_ok=True)
    for i, gen_sample in enumerate(gen_samples):
        distances = np.array([euclidean_distance(gen_sample, real_sample) for real_sample in real_data])
        closest_sample_idx = np.argmin(distances)
        closest_sample = real_data[closest_sample_idx]
        
        plt.figure(figsize=(10, 5))
        plt.plot(gen_sample, label='Generated Sample')
        plt.plot(closest_sample, label='Closest Real Sample', linestyle='--')
        plt.legend()
        plt.title(f'Sample {i+1}')
        plt.savefig(os.path.join(sample_dir, f'sample_{i+1}.png'))
        plt.close()

    np.save(os.path.join(sample_dir, 'generated_samples.npy'), gen_samples)

if __name__ == '__main__':
    # Load real data
    data = np.load('ca_data_99solar_15min.npy')
    data = data.reshape(-1, 96, 1)

    wgan = WGAN(data=data, seq_len=96, feature_dim=1, z_dim=100)

    model_path = 'Models/wgan_timeseries.ckpt'
    num_samples = 30
    z_sample = wgan.sample_z(num_samples)
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        gen_samples = sess.run(wgan.fake_data, feed_dict={wgan.z: z_sample})

    data_min = data.min()
    data_max = data.max()
    gen_samples = (gen_samples + 1) / 2 * (data_max - data_min) + data_min

    # Plot and save samples
    sample_dir = 'sample_wgan'
    plot_samples(gen_samples, data, sample_dir)
