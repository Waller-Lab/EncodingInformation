"""
Generate and save random samples from the L1 ball 
and from the space of bandlimited signals
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from encoding_information.gpu_utils import limit_gpu_memory_growth
limit_gpu_memory_growth()

from cleanplots import *
from tqdm import tqdm
from signal_utils_1D import *
from plot_utils_1D import *

import numpy as onp
import os


def generate_random_l1_samples(d, max_samples, batch_size):

    @partial(jax.jit, static_argnums=(1, 2))
    def random_l1_vector(key, N, d):
    # Generate a random radius and a random direction
        radius = jax.random.uniform(key, shape=(N, d,)) ** (1/d)
        key, subkey = jax.random.split(key)
        direction = jax.random.normal(subkey, shape=(N, d,))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)  # Normalize to unit length

        # Scale direction by radius and adjust to ensure all components are positive
        vectors = np.abs(radius * direction)
        mask = vectors.sum(axis=1) < 1
        return vectors, mask

    samples = []
    key = jax.random.PRNGKey(onp.random.randint(0, 1000000))
    i = 0
    if not os.path.exists(".cache"):
        os.mkdir(".cache")
    cache_filename = f".cache/{d}_dimensional_random_L1_ball_samples.npy"  # Name of the cache file
    # if it exists, load the samples from disk
    if os.path.exists(cache_filename):
        saved_samples = np.load(cache_filename, allow_pickle=True)
        print(f"Loaded {len(saved_samples)} samples from disk")
        samples.extend(saved_samples[:max_samples])

    while len(samples) < max_samples:
        vectors, mask = random_l1_vector(key, batch_size, d)
        valid_samples = vectors[mask]
        samples.extend(valid_samples)
        key, subkey = jax.random.split(key)
        print(i, len(samples), end='\r')
        i += 1
        if valid_samples.shape[0] == 0:
            continue
        samples = samples[:max_samples]

        # Save the valid samples to disk
        if os.path.exists(cache_filename):
            existing_samples = np.load(cache_filename, allow_pickle=True)
            combined_samples = np.concatenate([existing_samples, valid_samples], axis=0)
            np.save(cache_filename, combined_samples)
        else:
            np.save(cache_filename, valid_samples)

    return np.array(samples)


num_nyquist_samples_list = [2, 4, 6, 8, 10, 12]

delta_function = onp.zeros(OBJECT_LENGTH)
delta_function[delta_function.size // 2] = 1
input_signal = delta_function

threshold = 0.03


for d in num_nyquist_samples_list:
    print(f"Generating {d}-dimensional random hypersamosa samples")
    l1_ball_samples = generate_random_l1_samples(d, max_samples=2500, batch_size=int(1e7))


    cache_filename = f".cache/{d}_dimensional_random_hypersamosa_samples.npy"  # Name of the cache file
    hypersamosa_samples = []
    # if it exists
    if os.path.exists(cache_filename):
        # delete it
        os.remove(cache_filename)


    for sample in tqdm(l1_ball_samples):
        success = False
        output_signal = optimize_towards_target_signals([sample], input_signal, verbose=False)[1][0]
        distance = np.sqrt(np.sum((sample - output_signal) ** 2))
        if distance < threshold:
            success = True
            hypersamosa_samples.append(sample)
            if os.path.exists(cache_filename):
                os.remove(cache_filename)
            np.save(cache_filename, hypersamosa_samples)
    