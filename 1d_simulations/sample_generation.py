"""
Generate and save random samples from the L1 ball 
and from the space of bandlimited signals
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# this only works on startup!
from jax import config
config.update("jax_enable_x64", True)


from encoding_information.gpu_utils import limit_gpu_memory_growth
limit_gpu_memory_growth()

from cleanplots import *
from tqdm import tqdm
from signal_utils_1D import *
from plot_utils_1D import *

import numpy as onp
import os


def generate_random_l1_samples(d, max_samples, batch_size):
    print(f"Generating {max_samples} samples from the {d}-dimensional L1 ball\n")

    @partial(jax.jit, static_argnums=(1, 2))
    def random_l1_vector(key, N, d):
    # Generate a random radius and a random direction
        vectors = np.abs(jax.random.ball(key, d, p=1, shape=(N,)))
        return vectors, jax.random.split(key)[1]

    samples = []
    key = jax.random.PRNGKey(onp.random.randint(0, 1000000))
    i = 0
    if not os.path.exists(".cache"):
        os.mkdir(".cache")
    cache_filename = f".cache/{d}_dimensional_random_bandlimited_positive_samples.npy"  # Name of the cache file
    # if it exists, load the samples from disk
    if os.path.exists(cache_filename):
        saved_samples = np.load(cache_filename, allow_pickle=True)
        print(f"Loaded {len(saved_samples)} samples from disk")
        samples.extend(saved_samples[:max_samples])

    while len(samples) < max_samples:
        l1_ball_samples, key = random_l1_vector(key, batch_size, d)
        valid_samples = filter_l1_ball_samples_to_positive(l1_ball_samples=l1_ball_samples)
        samples.extend(valid_samples[:max_samples - len(samples)])
        print(i, len(samples), end='\r')
        i += 1
        if valid_samples.shape[0] == 0:
            continue
        samples = samples[:max_samples]

        # Save the valid samples to disk
        if os.path.exists(cache_filename):
            existing_samples = np.load(cache_filename, allow_pickle=True)
            combined_samples = onp.concatenate([existing_samples, onp.array(samples)], axis=0)
            combined_samples = combined_samples[:max_samples]
            np.save(cache_filename, combined_samples)
        else:
            np.save(cache_filename, onp.array(samples))

    return np.array(samples)


num_nyquist_samples_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]



for d in num_nyquist_samples_list:
    l1_ball_samples = generate_random_l1_samples(d, max_samples=3000, batch_size=int(1e5))
