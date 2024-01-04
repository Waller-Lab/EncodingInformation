"""
Utility functions ad classes for working with tensorflow/keras
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.gridspec as gridspec
import os
import shutil
import warnings
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers



    
def _gather_dataset(image_target_generator):
    #read first one to infer size
    for an_image, a_target in image_target_generator():
        if an_image is None:
            warnings.warn('generator producing None iamge')
            continue # special case needed when excluding some data from training
        break
        
    output_signature=(
                     tf.TensorSpec(shape=an_image.shape, dtype=tf.float32), 
                tf.TensorSpec(a_target.shape, dtype=tf.float32))
    all_dataset = tf.data.Dataset.from_generator(image_target_generator, 
                                             output_signature=output_signature)
    all_dataset = all_dataset.prefetch(8)

    return all_dataset
    # if size is None:
    #     #iterate through data to calculate size if we don't know it already
    #     size = 0
    #     for example in all_dataset:
    #         size += 1
    # return all_dataset, size

def prepare_test_dataset(test_fraction, image_target_generator, size, **kwargs):
    """
    Prepare test dataset from tfrecord
    """
    all_dataset = _gather_dataset(image_target_generator)
    
    test_size = int(test_fraction * size)
    # if return_size:
    return all_dataset.take(test_size), test_size
    # return all_dataset.take(test_size)
            
def prepare_datasets(validation_fraction=None, image_target_generator=None, size=None, 
                     batch_size=None, test_fraction=0, 
                     augment=None, repeat=True, filter_fn=None,
                     **kwargs):
    """
    Prepare TensorFlow Dataset for training
    """
    if size is None or image_target_generator is None or validation_fraction is None:
        raise Exception('invalid params')
    
    all_dataset = _gather_dataset(image_target_generator)

    validation_size = int(validation_fraction * size)
    print('Validation set size: {}'.format(validation_size))

    test_size = int(test_fraction * size)

    non_test_data = all_dataset.skip(test_size)
    train_dataset = non_test_data.skip(validation_size).repeat() if repeat else non_test_data.skip(validation_size)
    validation_dataset = non_test_data.take(validation_size).repeat() if repeat else non_test_data.take(validation_size)

    # if augment is not None:
    #     #TODO:
    #     pass
    
    if filter_fn is not None:
        # This can be used to exclude certain data after dividing into test/train/val
        # For example to include only examples with a certain targer type
        train_dataset = train_dataset.filter(filter_fn)
        validation_dataset = validation_dataset.filter(filter_fn)

    if batch_size is not None:
        validation_dataset = validation_dataset.batch(batch_size)
        train_dataset = train_dataset.batch(batch_size)
        #training on TPU requires defined shape
        val_steps = validation_size // batch_size
        
        train_dataset = train_dataset.prefetch(8)
        validation_dataset = validation_dataset.prefetch(8)
        
        return train_dataset, validation_dataset, val_steps
    else:
        train_dataset = train_dataset.prefetch(8)
        validation_dataset = validation_dataset.prefetch(8)
        
        return train_dataset, validation_dataset

    #     def augment(image, target):
#         """
#         Could do data augmentation here, if desired
#         """
#         image = tf.image.flip_up_down(image)
#         image = tf.image.flip_left_right(image)
#         image = tf.image.transpose(image)
#         return image, target


    
def compute_mean_sd(dataset, size=100):
        
    images = []
    for i, _ in dataset:
        images.append(i)
        batch_size = i.shape[0]
        if len(images) * batch_size > size:
            break
    images = np.concatenate(images, axis=0)
    
    means = np.mean(images, axis=(0, 1, 2))
    stddevs = np.std(images, axis=(0, 1, 2))
    return means, stddevs
        
    




def _parse_function(example_proto):
    """
    Function for unpacking images from TFRecord dataset
    """
    feature_description = {
        'channels': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }

    # Parse the input `tf.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_description)
    image_tensor = tf.io.decode_raw(example['image_raw'], tf.float32)
    image = tf.reshape(image_tensor, [example['height'], example['width'], example['channels']])
    target = tf.io.decode_raw(example['target'], tf.float32)
    
#     return image, image #autoencoder, so return image twice for input and target
    return image, target


"""
Convenience functions for tf records which no longer matter
"""


def convert_to_tfrecord(bsccm, tf_record_root, batch=1,
                        markers=None,
                        channels=(None, 'DF_50', 'DF_75'), 
                        contrast_types=('dpc', 'led_array', 'led_array'),
                       max_examples=None, ch_filename=None, dim=128):
    """
    Take BSCCM data and extract the batch and channels requested, preparing a TFRecord
    dataset with these channels as inputs and the target marker as output.
    """

    
    # Open data
    all_indices = bsccm.get_indices(batch=0, antibodies=markers)
    
    two_spectra_model_names, two_spectra_data, four_spectra_model_names, four_spectra_data = bsccm.get_surface_marker_data(all_indices)

    # Reorder the columns to be consistent with supplied order of markers
    # TODO: swap in 4 marker model here
    columns = []
    for marker in markers:
        col_index = np.flatnonzero([marker in name for name in two_spectra_model_names])[0]
        columns.append(two_spectra_data[:, col_index])
    target_mat = np.stack(columns, axis=1)    
    
    
    if max_examples is None:
        max_examples = all_indices.size
        
    #Autogenerate name for file like Batch-1_Xdpc-DF_50-DF_75X_size-12345.tfrecord
    channel_names = [channel if channel is not None else ct for (channel, ct) in zip(channels, contrast_types) ]
    tf_record_path = tf_record_root + 'Batch-{}_X{}X_size-{}.tfrecord'.format(batch, 
                                                    '-'.join(channel_names) if ch_filename is None else ch_filename, max_examples)

    #shuffle data before resaving
    shuffle_indices = np.arange(all_indices.size)
    np.random.seed(123456)
    np.random.shuffle(shuffle_indices)
    model_targets = target_mat[shuffle_indices]
    model_image_indices = all_indices[shuffle_indices]

    #generator funciton for loading one image at a time
    def image_target_generator():
        for index, target in zip(model_image_indices, model_targets):
            multi_channel_img = np.stack([bsccm.read_image(index, contrast_type=contrast, channel=ch).astype(np.float32)
                     for contrast, ch in zip(contrast_types, channels)], axis=-1)
            # crops to central region of image
            if dim == 128:
                image_cropped = multi_channel_img
            else:                
                edge_pad = (128 - dim) // 2
                image_cropped = multi_channel_img[edge_pad: -edge_pad, edge_pad: -edge_pad]
            yield image_cropped, target.astype(np.float32)

    print('Total size: {}'.format(max_examples))
    _write_tfrecord(tf_record_path, image_target_generator(), max_number=max_examples)   
    print('markers', markers)
    


#### Functions for I/O of binary data using TFRecord format ####

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _write_tfrecord(path, generator, max_number=None):
    """
    Write data to TFRecord file for fast training
    
    """
        
    writer = tf.io.TFRecordWriter(path)
    for count, (img, target) in enumerate(generator):
        print('Writing image {}\r'.format(count), end='')

        if max_number is not None and count == max_number:
            break
        
        channels = img.shape[2]
        height = img.shape[1]
        width = img.shape[0]
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'channels': _int64_feature(channels),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img.tostring()),
            'target': _bytes_feature(target.tostring()),
        }))

        writer.write(example.SerializeToString())

    writer.close()
    print('Written to: ' + path)

