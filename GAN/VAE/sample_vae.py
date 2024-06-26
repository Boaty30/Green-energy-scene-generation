import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

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

    # Load VAE generator model
    generator = load_model('generator_time_series.h5')

    # Generate samples
    num_samples = 30
    latent_dim = 10  # Ensure this matches the latent_dim used in the VAE model
    z_sample = np.random.normal(size=(num_samples, latent_dim))
    gen_samples = generator.predict(z_sample)

    # Denormalize generated samples
    data_min = data.min()
    data_max = data.max()
    gen_samples = gen_samples * (data_max - data_min) + data_min

    # Plot and save samples
    sample_dir = 'sample_vae'
    plot_samples(gen_samples, data, sample_dir)
