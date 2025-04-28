import numpy as np # use regular numpy for now, simpler
import scipy
from tqdm import tqdm
# import tensorflow as tf
# import tensorflow.keras as tfk
import gc
import warnings

import skimage
import skimage.io
from skimage.transform import resize

# from tensorflow.keras.optimizers import SGD

def tile_9_images(data_set):
    # takes 9 images and forms a tiled image
    assert len(data_set) == 9
    return np.block([[data_set[0], data_set[1], data_set[2]],[data_set[3], data_set[4], data_set[5]],[data_set[6], data_set[7], data_set[8]]])

def generate_random_tiled_data(x_set, y_set, seed_value=-1):
    # takes a set of images and labels and returns a set of tiled images and corresponding labels
    # the size of the output should be 3x the size of the input
    vert_shape = x_set.shape[1] * 3
    horiz_shape = x_set.shape[2] * 3
    random_data = np.zeros((x_set.shape[0], vert_shape, horiz_shape)) # for mnist this was 84 x 84
    random_labels = np.zeros((y_set.shape[0], 1))
    if seed_value==-1:
        np.random.seed()
    else: 
        np.random.seed(seed_value)
    for i in range(x_set.shape[0]):
        img_items = np.random.choice(x_set.shape[0], size=9, replace=True)
        data_set = x_set[img_items]
        random_labels[i] = y_set[img_items[4]]
        random_data[i] = tile_9_images(data_set)
    return random_data, random_labels

def generate_repeated_tiled_data(x_set, y_set):
    # takes set of images and labels and returns a set of repeated tiled images and corresponding labels, no randomness
    # the size of the output is 3x the size of the input, this essentially is a wrapper for np.tile
    repeated_data = np.tile(x_set, (1, 3, 3))
    repeated_labels = y_set # the labels are just what they were
    return repeated_data, repeated_labels

def convolved_dataset(psf, random_tiled_data, verbose=True):
    # takes a psf and a set of tiled images and returns a set of convolved images, convolved image size is 2n + 1? same size as the random data when it's cropped
    # tile size is two images worth plus one extra index value at the 0th index
    vert_shape = random_tiled_data.shape[1] - psf.shape[0] + 1 #psf.shape[0] * 2 + 1
    horiz_shape = random_tiled_data.shape[2] - psf.shape[1] + 1 #psf.shape[1] * 2 + 1
    psf_dataset = np.zeros((random_tiled_data.shape[0], vert_shape, horiz_shape)) # 57 x 57 for the case of mnist 28x28 images, 65 x 65 for the cifar 32 x 32 images
    if verbose:
        iterator = tqdm(range(random_tiled_data.shape[0]), desc='Convolving images', unit='image')
    else:
        iterator = range(random_tiled_data.shape[0])
    for i in iterator:
        psf_dataset[i] = scipy.signal.fftconvolve(psf, random_tiled_data[i], mode='valid')
    return psf_dataset

def compute_entropy(eigenvalues):
    sum_log_evs = np.sum(np.log2(eigenvalues))
    D = eigenvalues.shape[0]
    gaussian_entropy = 0.5 * (sum_log_evs + D * np.log2(2 * np.pi * np.e))
    return gaussian_entropy

def add_shot_noise(photon_scaled_images, photon_fraction=None, photons_per_pixel=None, assume_noiseless=True, seed_value=-1):
    #adapted from main API, also uses a seed though
    if seed_value==-1:
        np.random.seed()
    else: 
        np.random.seed(seed_value)
    
    # check all pixels greater than 0
    if np.any(photon_scaled_images < 0):
        #warning about negative
        warnings.warn(f"Negative pixel values detected. Clipping to 0.")
        photon_scaled_images[photon_scaled_images < 0] = 0
    if photons_per_pixel is not None:
        if photons_per_pixel > np.mean(photon_scaled_images):
            warnings.warn(f"photons_per_pixel is greater than actual photon count ({photons_per_pixel}). Clipping to {np.mean(photon_scaled_images)}")
            photons_per_pixel = np.mean(photon_scaled_images)
        photon_fraction = photons_per_pixel / np.mean(photon_scaled_images)

    if photon_fraction > 1:
        warnings.warn(f"photon_fraction is greater than 1 ({photon_fraction}). Clipping to 1.")
        photon_fraction = 1

    if assume_noiseless:
        additional_sd = np.sqrt(photon_fraction * photon_scaled_images)
        if np.any(np.isnan(additional_sd)):
            warnings.warn('There are nans here')
            additional_sd[np.isnan(additional_sd)] = 0
        # something here goes weird for RML
        # 
    #else:
    #    additional_sd = np.sqrt(photon_fraction * photon_scaled_images) - photon_fraction * np.sqrt(photon_scaled_images)
    simulated_images = photon_scaled_images * photon_fraction + additional_sd * np.random.randn(*photon_scaled_images.shape)
    positive = np.array(simulated_images)
    positive[positive < 0] = 0 # cant have negative counts
    return np.array(positive)

def tf_cast(data):
    # normalizes data, loads it to a tensorflow array of type float32
    return tf.cast(data / np.max(data), tf.float32)
def tf_labels(labels):
    # loads labels to a tensorflow array of type int64
    return tf.cast(labels, tf.int64)



def run_model_simple(train_data, train_labels, test_data, test_labels, val_data, val_labels, seed_value=-1):
    if seed_value == -1:
        seed_val = np.random.randint(10, 1000)
        tfk.utils.set_random_seed(seed_val)
    else:
        tfk.utils.set_random_seed(seed_value)

    model = tfk.models.Sequential()
    model.add(tfk.layers.Flatten())
    model.add(tfk.layers.Dense(256, activation='relu'))
    model.add(tfk.layers.Dense(256, activation='relu'))
    model.add(tfk.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = tfk.callbacks.EarlyStopping(monitor="val_loss", # add in an early stopping option 
                                        mode="min", patience=5,
                                        restore_best_weights=True, verbose=1)
    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), batch_size=32, epochs=50, callbacks=[early_stop])
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    return history, model, test_loss, test_acc

def run_model_cnn(train_data, train_labels, test_data, test_labels, val_data, val_labels, seed_value=-1):
    # structure from https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist
    if seed_value == -1:
        seed_val = np.random.randint(10, 1000)
        tfk.utils.set_random_seed(seed_val)
    else:
        tfk.utils.set_random_seed(seed_value)

    model = tfk.models.Sequential()
    model.add(tfk.layers.Conv2D(64, kernel_size=5, padding='same', activation='relu', input_shape=(57, 57, 1))) #64 and 128 works very slightly better
    model.add(tfk.layers.MaxPool2D())
    model.add(tfk.layers.Conv2D(128, kernel_size=5, padding='same', activation='relu'))
    model.add(tfk.layers.MaxPool2D())
    #model.add(tfk.layers.Conv2D(64, kernel_size=5, padding='same', activation='relu'))
    #model.add(tfk.layers.MaxPool2D(padding='same'))
    model.add(tfk.layers.Flatten())

    #model.add(tfk.layers.Dense(256, activation='relu'))
    model.add(tfk.layers.Dense(128, activation='relu'))
    model.add(tfk.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = tfk.callbacks.EarlyStopping(monitor="val_loss", # add in an early stopping option 
                                        mode="min", patience=5,
                                        restore_best_weights=True, verbose=1)
    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=50, batch_size=32, callbacks=[early_stop]) #validation data is not test data
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    return history, model, test_loss, test_acc

def seeded_permutation(seed_value, n):
    # given fixed seed returns permutation order
    np.random.seed(seed_value)
    permutation_order = np.random.permutation(n)
    return permutation_order

def segmented_indices(permutation_order, n, training_fraction, test_fraction):
    #given permutation order returns indices for each of the three sets
    training_indices = permutation_order[:int(training_fraction*n)]
    test_indices = permutation_order[int(training_fraction*n):int((training_fraction+test_fraction)*n)]
    validation_indices = permutation_order[int((training_fraction+test_fraction)*n):]
    return training_indices, test_indices, validation_indices

def permute_data(data, labels, seed_value, training_fraction=0.8, test_fraction=0.1): 
    #validation fraction is implicit, if including a validation set, expect to use the remaining fraction of the data
    permutation_order = seeded_permutation(seed_value, data.shape[0])
    training_indices, test_indices, validation_indices = segmented_indices(permutation_order, data.shape[0], training_fraction, test_fraction)

    training_data = data[training_indices]
    training_labels = labels[training_indices]
    testing_data = data[test_indices]
    testing_labels = labels[test_indices]
    validation_data = data[validation_indices]
    validation_labels = labels[validation_indices]

    return (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels)

def add_gaussian_noise(data, noise_level, seed_value=-1):
    if seed_value==-1:
        np.random.seed()
    else: 
        np.random.seed(seed_value)
    return data + noise_level * np.random.randn(*data.shape)

def confidence_bars(data_array, noise_length, confidence_interval=0.95):
    # can also use confidence interval 0.9 or 0.99 if want slightly different bounds
    error_lo = np.percentile(data_array, 100 * (1 - confidence_interval) / 2, axis=1)
    error_hi = np.percentile(data_array, 100 * (1 - (1 - confidence_interval) / 2), axis=1)
    mean = np.mean(data_array, axis=1)
    assert len(error_lo) == len(mean) == len(error_hi) == noise_length
    return error_lo, error_hi, mean


######### This function is very outdated, don't use it!! used to be called test_system use the ones below instead
#########
def test_system_old(noise_level, psf_name, model_name, seed_values, data, labels, training_fraction, testing_fraction,  diffuser_region, phlat_region, psf, noise_type, rml_region):
    # runs the model for the number of seeds given, returns the test accuracy for each seed
    test_accuracy_list = []
    for seed_value in seed_values:
        seed_value = int(seed_value)
        tfk.backend.clear_session()
        gc.collect()
        tfk.utils.set_random_seed(seed_value) # set random seed out here too?
        training, testing, validation = permute_data(data, labels, seed_value, training_fraction, testing_fraction)
        x_train, y_train = training
        x_test, y_test = testing
        x_validation, y_validation = validation

        random_test_data, random_test_labels = generate_random_tiled_data(x_test, y_test, seed_value)
        random_train_data, random_train_labels = generate_random_tiled_data(x_train, y_train, seed_value)
        random_valid_data, random_valid_labels = generate_random_tiled_data(x_validation, y_validation, seed_value)

        if psf_name == 'uc':
            test_data = random_test_data[:, 14:-13, 14:-13]
            train_data = random_train_data[:, 14:-13, 14:-13]
            valid_data = random_valid_data[:, 14:-13, 14:-13]
        if psf_name == 'psf_4':
            test_data = convolved_dataset(psf, random_test_data)
            train_data = convolved_dataset(psf, random_train_data)
            valid_data = convolved_dataset(psf, random_valid_data)
        if psf_name == 'diffuser':
            test_data = convolved_dataset(diffuser_region, random_test_data)
            train_data = convolved_dataset(diffuser_region, random_train_data)
            valid_data = convolved_dataset(diffuser_region, random_valid_data)  
        if psf_name == 'phlat':
            test_data = convolved_dataset(phlat_region, random_test_data)
            train_data = convolved_dataset(phlat_region, random_train_data)
            valid_data = convolved_dataset(phlat_region, random_valid_data)
        # 6/19/23 added RML option
        if psf_name == 'rml':
            test_data = convolved_dataset(rml_region, random_test_data)
            train_data = convolved_dataset(rml_region, random_train_data)
            valid_data = convolved_dataset(rml_region, random_valid_data)

        # address any tiny floating point negative values, which only occur in RML data
        if np.any(test_data < 0):
            #print('negative values in test data for {} psf'.format(psf_name))
            test_data[test_data < 0] = 0
        if np.any(train_data < 0):
            #print('negative values in train data for {} psf'.format(psf_name))
            train_data[train_data < 0] = 0
        if np.any(valid_data < 0):
            #print('negative values in valid data for {} psf'.format(psf_name))
            valid_data[valid_data < 0] = 0
            

        # additive gaussian noise, add noise after convolving, fixed 5/15/2023
        if noise_type == 'gaussian':
            test_data = add_gaussian_noise(test_data, noise_level, seed_value)
            train_data = add_gaussian_noise(train_data, noise_level, seed_value)
            valid_data = add_gaussian_noise(valid_data, noise_level, seed_value)
        if noise_type == 'poisson':
            test_data = add_shot_noise(test_data, photons_per_pixel=noise_level, seed_value=seed_value, assume_noiseless=True)
            train_data = add_shot_noise(train_data, photons_per_pixel=noise_level, seed_value=seed_value, assume_noiseless=True)
            valid_data = add_shot_noise(valid_data, photons_per_pixel=noise_level, seed_value=seed_value, assume_noiseless=True)

        train_data, test_data, valid_data = tf_cast(train_data), tf_cast(test_data), tf_cast(valid_data)
        random_train_labels, random_test_labels, random_valid_labels = tf_labels(random_train_labels), tf_labels(random_test_labels), tf_labels(random_valid_labels)
        
        if model_name == 'simple':
            history, model, test_loss, test_acc = run_model_simple(train_data, random_train_labels, test_data, random_test_labels, valid_data, random_valid_labels, seed_value)
        if model_name == 'cnn':
            history, model, test_loss, test_acc = run_model_cnn(train_data, random_train_labels, test_data, random_test_labels, valid_data, random_valid_labels, seed_value)
        test_accuracy_list.append(test_acc)
    np.save('classification_results_rml_psf_619/test_accuracy_{}_noise_{}_{}_psf_{}_model.npy'.format(noise_level, noise_type, psf_name, model_name), test_accuracy_list)

 ###### CNN for 32x32 CIFAR10 images 
    # Originally written 11/14/2023, but then lost in a merge, recopied 1/14/2024
def run_model_cnn_cifar(train_data, train_labels, test_data, test_labels, val_data, val_labels, seed_value=-1, max_epochs=50, patience=5):
    # structure from https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist
    # default architecture is 50 epochs and patience 5, but recently some need longer patience
    if seed_value == -1:
        seed_val = np.random.randint(10, 1000)
        tfk.utils.set_random_seed(seed_val)
    else:
        tfk.utils.set_random_seed(seed_value)
    model = tfk.models.Sequential()
    model.add(tfk.layers.Conv2D(64, kernel_size=5, padding='same', activation='relu', input_shape=(65, 65, 1)))
    model.add(tfk.layers.MaxPool2D())
    model.add(tfk.layers.Conv2D(128, kernel_size=5, padding='same', activation='relu'))
    model.add(tfk.layers.MaxPool2D())
    #model.add(tfk.layers.Conv2D(64, kernel_size=5, padding='same', activation='relu'))
    #model.add(tfk.layers.MaxPool2D(padding='same'))
    model.add(tfk.layers.Flatten())

    #model.add(tfk.layers.Dense(256, activation='relu'))
    model.add(tfk.layers.Dense(128, activation='relu'))
    model.add(tfk.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = tfk.callbacks.EarlyStopping(monitor="val_loss", # add in an early stopping option 
                                        mode="min", patience=patience,
                                        restore_best_weights=True, verbose=1)
    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=max_epochs, batch_size=32, callbacks=[early_stop]) #validation data is not test data
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    return history, model, test_loss, test_acc

def make_ttv_sets(data, labels, seed_value, training_fraction, testing_fraction):
    training, testing, validation = permute_data(data, labels, seed_value, training_fraction, testing_fraction)
    training_data, training_labels = training
    testing_data, testing_labels = testing
    validation_data, validation_labels = validation
    training_data, testing_data, validation_data = tf_cast(training_data), tf_cast(testing_data), tf_cast(validation_data)
    training_labels, testing_labels, validation_labels = tf_labels(training_labels), tf_labels(testing_labels), tf_labels(validation_labels)
    return (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels)

def run_network_cifar(data, labels, seed_value, training_fraction, testing_fraction, mode='cnn', max_epochs=50, patience=5):
    # small modification to be able to run 32x32 image data 
    training, testing, validation = make_ttv_sets(data, labels, seed_value, training_fraction, testing_fraction)
    if mode == 'cnn':
        history, model, test_loss, test_acc = run_model_cnn_cifar(training[0], training[1],
                                                            testing[0], testing[1],
                                                            validation[0], validation[1], seed_value, max_epochs, patience)
    elif mode == 'simple':
        history, model, test_loss, test_acc = run_model_simple(training[0], training[1], 
                                                                     testing[0], testing[1],
                                                                     validation[0], validation[1], seed_value)
    elif mode == 'new_cnn':
        history, model, test_loss, test_acc = current_testing_model(training[0], training[1],
                                                                    testing[0], testing[1], 
                                                                    validation[0], validation[1], seed_value, max_epochs, patience)
    elif mode == 'mom_cnn':
        history, model, test_loss, test_acc = momentum_testing_model(training[0], training[1],
                                                                    testing[0], testing[1], 
                                                                    validation[0], validation[1], seed_value, max_epochs, patience)
    return history, model, test_loss, test_acc  


def load_diffuser_psf():
    diffuser_psf = skimage.io.imread('psfs/diffuser_psf.png')
    diffuser_psf = diffuser_psf[:,:,1]
    diffuser_resize = diffuser_psf[200:500, 250:550]
    diffuser_resize = resize(diffuser_resize, (100, 100), anti_aliasing=True)  #resize(diffuser_psf, (28, 28))
    diffuser_region = diffuser_resize[:28, :28]
    diffuser_region /=  np.sum(diffuser_region)
    return diffuser_region

def load_phlat_psf():
    phlat_psf = skimage.io.imread('psfs/phlat_psf.png')
    phlat_psf = phlat_psf[900:2900, 1500:3500, 1]
    phlat_psf = resize(phlat_psf, (200, 200), anti_aliasing=True)
    phlat_region = phlat_psf[10:38, 20:48]
    phlat_region /= np.sum(phlat_region)
    return phlat_region

def load_4_psf():
    psf = np.zeros((28, 28))
    psf[20,20] = 1
    psf[15, 10] = 1
    psf[5, 13] = 1
    psf[23, 6] = 1
    psf = scipy.ndimage.gaussian_filter(psf, sigma=1)
    psf /= np.sum(psf)
    return psf

# 6/9/23 added rml option
def load_rml_psf():
    rml_psf = skimage.io.imread('psfs/psf_8holes.png')
    rml_psf = rml_psf[1000:3000, 1500:3500]
    rml_psf_resize = resize(rml_psf, (100, 100), anti_aliasing=True)
    rml_psf_region = rml_psf_resize[40:100, :60]
    rml_psf_region = resize(rml_psf_region, (28, 28), anti_aliasing=True)
    rml_psf_region /= np.sum(rml_psf_region)
    return rml_psf_region

def load_rml_new_psf():
    rml_psf = skimage.io.imread('psfs/psf_8holes.png')
    rml_psf = rml_psf[1000:3000, 1500:3500]
    rml_psf_small = resize(rml_psf, (85, 85), anti_aliasing=True)
    rml_psf_region = rml_psf_small[52:80, 10:38]
    rml_psf_region /= np.sum(rml_psf_region)
    return rml_psf_region 

def load_single_lens():
    one_lens = np.zeros((28, 28))
    one_lens[14, 14] = 1
    one_lens = scipy.ndimage.gaussian_filter(one_lens, sigma=0.8)
    one_lens /= np.sum(one_lens)
    return one_lens

def load_two_lens():
    two_lens = np.zeros((28, 28))
    two_lens[10, 10] = 1
    two_lens[20, 20] = 1
    two_lens = scipy.ndimage.gaussian_filter(two_lens, sigma=0.8)
    two_lens /= np.sum(two_lens)
    return two_lens

def load_three_lens():
    three_lens = np.zeros((28, 28))
    three_lens[8, 12] = 1 
    three_lens[16, 20] = 1
    three_lens[20, 7] = 1
    three_lens = scipy.ndimage.gaussian_filter(three_lens, sigma=0.8)
    three_lens /= np.sum(three_lens)
    return three_lens


def load_single_lens_32():
    one_lens = np.zeros((32, 32))
    one_lens[16, 16] = 1
    one_lens = scipy.ndimage.gaussian_filter(one_lens, sigma=0.8)
    one_lens /= np.sum(one_lens)
    return one_lens

def load_two_lens_32():
    two_lens = np.zeros((32, 32))
    two_lens[10, 10] = 1
    two_lens[21, 21] = 1
    two_lens = scipy.ndimage.gaussian_filter(two_lens, sigma=0.8)
    two_lens /= np.sum(two_lens)
    return two_lens

def load_three_lens_32():
    three_lens = np.zeros((32, 32))
    three_lens[9, 12] = 1
    three_lens[17, 22] = 1
    three_lens[24, 8] = 1
    three_lens = scipy.ndimage.gaussian_filter(three_lens, sigma=0.8)
    three_lens /= np.sum(three_lens)
    return three_lens

def load_four_lens_32():
    psf = np.zeros((32, 32))
    psf[22, 22] = 1
    psf[15, 10] = 1
    psf[5, 12] = 1
    psf[28, 8] = 1
    psf = scipy.ndimage.gaussian_filter(psf, sigma=1) # note that this one is sigma 1, for mnist it's sigma 0.8
    psf /= np.sum(psf)
    return psf

def load_diffuser_32():
    diffuser_psf = skimage.io.imread('/home/your_username/EncodingInformation/lensless_imager/psfs/diffuser_psf.png')
    diffuser_psf = diffuser_psf[:,:,1]
    diffuser_resize = diffuser_psf[200:500, 250:550]
    diffuser_resize = resize(diffuser_resize, (100, 100), anti_aliasing=True)  #resize(diffuser_psf, (28, 28))
    diffuser_region = diffuser_resize[:32, :32]
    diffuser_region /=  np.sum(diffuser_region)
    return diffuser_region



### 10/15/2023: Make new versions of the model functions that train with Datasets - first attempt failed

# lenses with centralized positions for use in task-specific estimations
def load_single_lens_uniform(size=32, sigma=0.8):
    one_lens = np.zeros((size, size))
    one_lens[16, 16] = 1
    one_lens = scipy.ndimage.gaussian_filter(one_lens, sigma=sigma)
    one_lens /= np.sum(one_lens)
    return one_lens

def load_two_lens_uniform(size=32, sigma=0.8):
    two_lens = np.zeros((size, size))
    two_lens[16, 16] = 1 
    two_lens[7, 9] = 1
    two_lens = scipy.ndimage.gaussian_filter(two_lens, sigma=sigma)
    two_lens /= np.sum(two_lens)
    return two_lens

def load_three_lens_uniform(size=32, sigma=0.8):
    three_lens = np.zeros((size, size))
    three_lens[16, 16] = 1
    three_lens[7, 9] = 1
    three_lens[23, 21] = 1
    three_lens = scipy.ndimage.gaussian_filter(three_lens, sigma=sigma)
    three_lens /= np.sum(three_lens)
    return three_lens

def load_four_lens_uniform(size=32, sigma=0.8):
    four_lens = np.zeros((size, size))
    four_lens[16, 16] = 1
    four_lens[7, 9] = 1
    four_lens[23, 21] = 1
    four_lens[8, 24] = 1
    four_lens = scipy.ndimage.gaussian_filter(four_lens, sigma=sigma)
    four_lens /= np.sum(four_lens)
    return four_lens
def load_five_lens_uniform(size=32, sigma=0.8):
    five_lens = np.zeros((size, size))
    five_lens[16, 16] = 1
    five_lens[7, 9] = 1
    five_lens[23, 21] = 1
    five_lens[8, 24] = 1
    five_lens[21, 5] = 1
    five_lens = scipy.ndimage.gaussian_filter(five_lens, sigma=sigma)
    five_lens /= np.sum(five_lens)
    return five_lens



## 01/24/2024 new CNN that's slightly deeper 
def current_testing_model(train_data, train_labels, test_data, test_labels, val_data, val_labels, seed_value=-1, max_epochs=50, patience=20):
    # structure from https://www.kaggle.com/code/amyjang/tensorflow-cifar10-cnn-tutorial 
    
    if seed_value == -1:
        seed_val = np.random.randint(10, 1000)
        tfk.utils.set_random_seed(seed_val)
    else:
        tfk.utils.set_random_seed(seed_value)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=5, padding='same', input_shape=(65, 65, 1), activation='relu'),
    tf.keras.layers.Conv2D(32, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
    ])

    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = tfk.callbacks.EarlyStopping(monitor="val_loss", # add in an early stopping option 
                                            mode="min", patience=patience,
                                            restore_best_weights=True, verbose=1)
    print(model.optimizer.get_config())

    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=max_epochs, batch_size=32, callbacks=[early_stop]) #validation data is not test data
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    return history, model, test_loss, test_acc




## 01/24/2024 new CNN that's slightly deeper 
def momentum_testing_model(train_data, train_labels, test_data, test_labels, val_data, val_labels, seed_value=-1, max_epochs=50, patience=20):
    # structure from https://www.kaggle.com/code/amyjang/tensorflow-cifar10-cnn-tutorial 
    # includes nesterov momentum feature, rather than regular momentum
    if seed_value == -1:
        seed_val = np.random.randint(10, 1000)
        tfk.utils.set_random_seed(seed_val)
    else:
        tfk.utils.set_random_seed(seed_value)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=5, padding='same', input_shape=(65, 65, 1), activation='relu'),
    tf.keras.layers.Conv2D(32, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
    ])

    model.compile(optimizer=SGD(momentum=0.9, nesterov=True), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = tfk.callbacks.EarlyStopping(monitor="val_loss", # add in an early stopping option 
                                            mode="min", patience=patience,
                                            restore_best_weights=True, verbose=1)
    
    print(model.optimizer.get_config())

    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=max_epochs, batch_size=32, callbacks=[early_stop]) #validation data is not test data
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    return history, model, test_loss, test_acc


# bootstrapping function 
def compute_bootstraps(mses, psnrs, ssims, test_set_length, num_bootstraps=100): 
    bootstrap_mses = []
    bootstrap_psnrs = []
    bootstrap_ssims = []
    for bootstrap_idx in tqdm(range(num_bootstraps), desc='Bootstrapping to compute confidence interval'):
        # select indices for sampling
        bootstrap_indices = np.random.choice(test_set_length, test_set_length, replace=True) 
        # take the metric values at those indices
        bootstrap_selected_mses = mses[bootstrap_indices]
        bootstrap_selected_psnrs = psnrs[bootstrap_indices]
        bootstrap_selected_ssims = ssims[bootstrap_indices]
        # accumulate the mean of the selected metric values
        bootstrap_mses.append(np.mean(bootstrap_selected_mses))
        bootstrap_psnrs.append(np.mean(bootstrap_selected_psnrs))
        bootstrap_ssims.append(np.mean(bootstrap_selected_ssims))
    bootstrap_mses = np.array(bootstrap_mses)
    bootstrap_psnrs = np.array(bootstrap_psnrs)
    bootstrap_ssims = np.array(bootstrap_ssims)
    return bootstrap_mses, bootstrap_psnrs, bootstrap_ssims

def compute_confidence_interval(list_of_items, confidence_interval=0.95):
    # use this one, final version
    assert confidence_interval > 0 and confidence_interval < 1
    mean_value = np.mean(list_of_items)
    lower_bound = np.percentile(list_of_items, 50 * (1 - confidence_interval))
    upper_bound = np.percentile(list_of_items, 50 * (1 + confidence_interval))
    return mean_value, lower_bound, upper_bound



def load_fake_rml(size=32, sigma=0.8):
    # make 6 lenslets, with variable blur for each one 
    fake_rml = np.zeros((size, size))
    fake_rml_2 = np.zeros((size, size))
    fake_rml_3 = np.zeros((size, size))
    fake_rml[16, 16] = 1 
    fake_rml_2[7, 9] = 1 
    fake_rml_2[23, 21] = 1
    fake_rml[8, 24] = 1
    fake_rml[21, 5] = 1
    fake_rml_3[26, 12] = 1
    fake_rml_3[3, 17] = 1
    fake_rml_2[16, 28] = 1

    
    fake_rml = scipy.ndimage.gaussian_filter(fake_rml, sigma=sigma)
    fake_rml_2 = scipy.ndimage.gaussian_filter(fake_rml_2, sigma=sigma*1.25)
    fake_rml_3 = scipy.ndimage.gaussian_filter(fake_rml_3, sigma=sigma*1.75)
    fake_rml += fake_rml_2
    fake_rml += fake_rml_3
    fake_rml /= np.sum(fake_rml)
    return fake_rml