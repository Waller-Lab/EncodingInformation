"""
Script that trains a single model based on the info in a provided config file
"""

print('~~~~~~~~~~~~~running train script~~~~~~~~~~~~~~')

from cookie_monster_backend_lib import train_script_setup, train_script_complete


config_file_path, saving_dir, config, hyperparameters, already_elapsed_time, \
    tensorboard_dir, logging_dir, model_dir, resume_backup_dir  = train_script_setup()


################################################
############# Model-specific training ##########
################################################

from pathlib import Path
from bsccm import BSCCM
import tensorflow as tf
import tensorflow.keras as tfk
from led_array.tf_util import prepare_datasets, compute_mean_sd
from encoding_information.bsccm_utils import get_bsccm_image_marker_generator
# from analysis.visualization import plot_density_network_output
from led_array.models.callbacks import DensityNetworkVisualizeCallback, ElapsedTimeCallback
from led_array.models.marker_models import build_marker_prediction_model, create_marker_prediction_loss
from pathlib import Path
import os
home = str(Path.home())


######## Load BSCCM ######
print('preparing datasets')
bsccm = BSCCM(config['data']['data_dir'] + '/{}/'.format(config['data']['dataset_name']), cache_index=True)
image_dim = bsccm.global_metadata['led_array']['image_shape'][0]
markers, image_target_generator, dataset_size, display_range, _ = get_bsccm_image_marker_generator(bsccm, image_dim=image_dim, **config["data"] )
##########################



###### Create callbacks #######
tfk.utils.set_random_seed(hyperparameters['seed'])

# Early stopping based on a particular marker rather than average of them all
if hyperparameters['single_marker_early_stopping']:
    monitor = 'val_{}{}_loss'.format( hyperparameters['single_marker_early_stopping'], 
                                     '_density' if hyperparameters['density_output'] else '')
    print('Monitoring {} instead of val_loss for early stopping'.format(monitor))
else:
    monitor = 'val_loss'

print('creating callbacks')
callbacks = [
    ElapsedTimeCallback(config, config_file_path, already_elapsed_time),
    tfk.callbacks.EarlyStopping(monitor=monitor, patience=hyperparameters['overshoot_epochs']),
    tfk.callbacks.ModelCheckpoint(filepath=model_dir + 'saved_model.h5', monitor=monitor, save_best_only=True),
    tfk.callbacks.TensorBoard(log_dir=tensorboard_dir)
]

if config['options']['resume_training']:
   callbacks.append(tf.keras.callbacks.BackupAndRestore(backup_dir=resume_backup_dir))
#########################



######## Prepare datasets and build model ###########
train_dataset, val_dataset, val_steps = prepare_datasets(
                        image_target_generator=image_target_generator, size=dataset_size,
                         **hyperparameters)

print('looking for existing model to erase or reload ', model_dir + 'saved_model.h5')
if os.path.exists(model_dir + 'saved_model.h5'):
    if not config['options']['resume_training']:
        raise Exception('Model already exists, but resume_training is not set to False')
    else:
        print('loading model to resume training')
        loss_fn = create_marker_prediction_loss(markers, **hyperparameters)
        model = tfk.models.load_model(model_dir + 'saved_model.h5', custom_objects={'loss': loss_fn})
        resumed = True
else:
    print('building new model')
    print('computing normalization')
    means, stddevs = compute_mean_sd(train_dataset, hyperparameters['num_examples_for_normalization'])
    print('building model')
    model = build_marker_prediction_model(means, stddevs, markers, image_dim=image_dim, **config['data'], **hyperparameters)
    resumed = False
    #################################################


###### Special desnity visualization callback 
if hyperparameters['density_output']:
    # create plots showing indivudual validation examples
    _, val_dataset_for_vis = prepare_datasets(image_target_generator=image_target_generator,
        **{key: hyperparameters[key] for key in hyperparameters.keys() if key != 'batch_size'},           
                        size=dataset_size, batch_size=None)
    cb = DensityNetworkVisualizeCallback(val_dataset_for_vis, markers, num_images=12,
                                                        logging_dir=logging_dir, display_range=display_range)
    if not resumed:
        print('making initial plot')
        cb.make_initial_plot(model)
        print('initial plot complete')
    callbacks += [cb]

    
if hyperparameters['all_marker_training']:
    history = model.fit(train_dataset, validation_data=val_dataset, validation_steps=val_steps,
                   callbacks=callbacks, 
              epochs=hyperparameters['max_epochs'], steps_per_epoch=hyperparameters['steps_per_epoch'])
print('all marker training complete')

if hyperparameters['single_marker_training']:
    marker = hyperparameters['single_marker_training']
    marker_index = markers.index(marker)

    def filter_to_marker(image, target):
        return not tf.math.is_nan(target[marker_index])

    # Regenerate dataset to filter only to ones with that marker
    markers, image_target_generator, dataset_size, display_range, _ = get_bsccm_image_marker_generator(
        bsccm, image_dim=image_dim, **config["data"], single_marker=hyperparameters['single_marker_training'])
    train_dataset, val_dataset, val_steps = prepare_datasets(
                        image_target_generator=image_target_generator, size=dataset_size,
                        filter_fn=filter_to_marker,
                         **hyperparameters)
    
    print('starting single marker_training')
    num_previous_steps = len(history.history['loss'])
    model.fit(train_dataset, validation_data=val_dataset, validation_steps=val_steps,
               callbacks=callbacks,  epochs=hyperparameters['max_epochs'],
                 steps_per_epoch=hyperparameters['steps_per_epoch'], initial_epoch=num_previous_steps)
    
    
print('Training complete')


#######################################################
##### Training complete file flag for scheduler #######
#######################################################
train_script_complete(saving_dir)

print('~~~~~~~~~~~~~train script complete~~~~~~~~~~~~~~')