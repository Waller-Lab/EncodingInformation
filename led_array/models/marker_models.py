"""
Code for creating various models in TF/Keras
"""
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

def build_featurizer_network(arch, new_num_channels, image_dim, change_input_channels):
    """
    Want to use a model pretrained on imagenet, but want to use a different number than
    3 input channels. So this function will averge the first convolutional layer weights
    over the RGB channels and replicate that across an arbitrary number of channels

    The untrained network here actually gets all its weights reset to the pretrained network,
    so it is not really untrained. 
    """
    if arch in ['DN121', 'DenseNet121']:
        net_pretrained = tfk.applications.densenet.DenseNet121(input_shape=(image_dim, image_dim, 3),
                                        include_top=False, weights='imagenet',
                                        input_tensor=None, pooling=None)
        net_untrained = tfk.applications.densenet.DenseNet121(input_shape=(image_dim, image_dim, new_num_channels),
                                        include_top=False, weights=None,
                                        input_tensor=None, pooling=None)
    elif arch == 'DN201':
        net_pretrained = tfk.applications.densenet.DenseNet201(input_shape=(image_dim, image_dim, 3),
                                include_top=False, weights='imagenet',
                                input_tensor=None, pooling=None)
        net_untrained = tfk.applications.densenet.DenseNet201(input_shape=(image_dim, image_dim, new_num_channels),
                                        include_top=False, weights=None,
                                        input_tensor=None, pooling=None)
    elif arch == 'VGG16':
        net_pretrained = tfk.applications.VGG16(input_shape=(image_dim, image_dim, 3),
                                                include_top=False, weights='imagenet',
                                                input_tensor=None, pooling=None)
        net_untrained = tfk.applications.VGG16(input_shape=(image_dim, image_dim, new_num_channels),
                                        include_top=False, weights=None,
                                        input_tensor=None, pooling=None)
    elif arch == 'EfficientNetV2S':
        net_pretrained = tfk.applications.efficientnet_v2.EfficientNetV2S(input_shape=(image_dim, image_dim, 3),
                                include_top=False, weights='imagenet',
                                input_tensor=None, pooling=None)
        net_untrained = tfk.applications.efficientnet_v2.EfficientNetV2S(input_shape=(image_dim, image_dim, new_num_channels),
                                        include_top=False, weights=None,
                                        input_tensor=None, pooling=None)
    elif arch == 'EfficientNetV2L':
        net_pretrained = tfk.applications.efficientnet_v2.EfficientNetV2L(input_shape=(image_dim, image_dim, 3),
                                include_top=False, weights='imagenet',
                                input_tensor=None, pooling=None)
        net_untrained = tfk.applications.efficientnet_v2.EfficientNetV2L(input_shape=(image_dim, image_dim, new_num_channels),
                                        include_top=False, weights=None,
                                        input_tensor=None, pooling=None)
    elif arch == 'ConvNextTiny':
        net_pretrained = tfk.applications.convnext.ConvNeXtTiny(input_shape=(image_dim, image_dim, 3),
                                include_top=False, weights='imagenet',
                                input_tensor=None, pooling=None)
        net_untrained = tfk.applications.convnext.ConvNeXtTiny(input_shape=(image_dim, image_dim, new_num_channels),
                                        include_top=False, weights=None,
                                        input_tensor=None, pooling=None)
    elif arch == 'ConvNextXLarge':
        net_pretrained = tfk.applications.convnext.ConvNeXtXLarge(input_shape=(image_dim, image_dim, 3),
                                include_top=False, weights='imagenet',
                                input_tensor=None, pooling=None)
        net_untrained = tfk.applications.convnext.ConvNeXtXLarge(input_shape=(image_dim, image_dim, new_num_channels),
                                        include_top=False, weights=None,
                                        input_tensor=None, pooling=None)
    else:
        raise Exception('Unknown arch')
        

    pretrained_layers = net_pretrained.layers
    if 'ConvNext' in arch:
        # remove the second element of the list, because it is a preprocessing layer to ignore
        pretrained_layers.pop(1)
    
    for i, layer in enumerate(net_untrained.layers):
        if i == 1 and change_input_channels and 'ConvNext' in arch:
            kernel, bias = pretrained_layers[i].layers[0].get_weights()
            new_kernel = np.stack(new_num_channels * [np.mean(kernel, axis=2)], axis=2)
            layer.layers[0].set_weights([new_kernel, bias])
        elif i == 2 and change_input_channels and 'ConvNext' not in arch:
            kernel = pretrained_layers[i].get_weights()[0]
            #replicate first channel weigth across all channels
            new_kernel = np.stack(new_num_channels * [np.mean(kernel, axis=2)], axis=2)
            layer.set_weights([new_kernel])
        else:
            layer.set_weights(pretrained_layers[i].get_weights())
    return net_untrained
        
    
def build_marker_prediction_model(means, stddevs, markers, image_dim, change_input_channels=True,
                                  arch=None, density_output=True, num_mixture_components=-1, 
                                  mixture_type=None, **kwargs):
    """
    Build a set models for predicting multiple markers. They share a common CNN encoder,
    but have their own distinct fully connected layers for predicting each marker
    """
    inputs = tfk.Input(shape = (image_dim, image_dim, means.size))
    preprocess = tfkl.Lambda(lambda x: (x - means) / stddevs)(inputs)

    network = build_featurizer_network(arch, means.size, image_dim, change_input_channels=change_input_channels)
    features = network(preprocess)

    pooled = tfk.layers.GlobalAveragePooling2D()(features)
    flat_features = tfk.layers.Flatten()(pooled)

    outputs = {}
    for i in range(len(markers)):
        l = tfkl.Dropout(0.5)(flat_features)
        l = tfkl.Dense(units=400, activation='tanh')(l)
        l = tfkl.Dropout(0.5)(l)
        l = tfkl.Dense(units=400, activation='tanh')(l)
            
        # Add point or distribution prediction
        if not density_output:
             outputs[markers[i]] = tfkl.Dense(units=1, activation=None, name=markers[i])(l)
        else:
            mu = tfkl.Dense(units=num_mixture_components, activation=None)(l)
            if mixture_type == 'gaussian_fixed_sigma':
                sigma = Sigma(num_mixture_components)()
            else:
                sigma = tfkl.Dense(units=num_mixture_components, activation='softplus')(l)
            alpha = tfkl.Dense(units=num_mixture_components, activation='softmax')(l)
            
            if 'gaussian' in mixture_type:
                mixture = tfpl.DistributionLambda(
                    make_distribution_fn=lambda params:
                    tfd.MixtureSameFamily(
                        mixture_distribution=tfd.Categorical( probs=params[0]),
                        components_distribution=tfd.Normal(loc=params[1], scale=params[2])),
                            convert_to_tensor_fn=tfd.Distribution.sample,
                            name=markers[i] + '_density')([alpha, mu, sigma])
            elif mixture_type == 'logistic':
                mixture = tfpl.DistributionLambda(
                    make_distribution_fn=lambda params:
                    tfd.MixtureSameFamily(
                        mixture_distribution=tfd.Categorical( probs=params[0]),
                        components_distribution=tfd.Logistic(loc=params[1], scale=params[2])),
                            convert_to_tensor_fn=tfd.Distribution.sample,
                            name=markers[i] + '_density')([alpha, mu, sigma])
            else:
                raise Exception('Unknown mixture type')

            outputs[markers[i]] = mixture
        
        
    model = tfk.Model(inputs, outputs)

    losses = create_marker_prediction_loss(markers, density_output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs['learning_rate'], beta_1=0.9, beta_2=0.999, amsgrad=True)
    # Removed for now, do with filtered data instead: loss_weights=kwargs['loss_weights']
    model.compile(optimizer=optimizer, loss=losses)  

    return model

def create_marker_prediction_loss(markers, density_output, **kwargs):
    # Define custom loss functions that mask out nans for missing target data
    losses = {}
    for loss_index in range(len(markers)):
        if not density_output:            
            # Pass in loss_index as loss_index to avoid late binding bug (https://stackoverflow.com/questions/3431676/creating-functions-in-a-loop)
            def loss (y_true, y_pred, loss_index=loss_index):
                #loss in MSE 
                marker_true = y_true[:, loss_index]
                # if its zero that means it doesnt have this marker
                mask = tf.cast(tf.logical_not(tf.math.is_nan(marker_true)), tf.bool)

                true_masked = tf.boolean_mask(tf.squeeze(marker_true), mask)
                pred_masked = tf.boolean_mask(tf.squeeze(y_pred), mask)
                sq_diff = tf.square(true_masked - pred_masked)

                #Route through a cond in case mask is all false to avoid a nan loss
                l = tf.cond(tf.reduce_any(mask), lambda : tf.reduce_sum(sq_diff, axis=-1), 
                            lambda : tf.convert_to_tensor(np.array(0, dtype=np.float32)))
                return l    
        else:
            #loss is negative log likelihood of mixture distribution
            def loss(y_true, y_pred, loss_index=loss_index):
                #Get the target value for the output corresponding to the marker we're looking at
                marker_true = y_true[:, loss_index]
                # create mask for nan values (i.e. no target recorded)
                mask = tf.cast(tf.logical_not(tf.math.is_nan(marker_true)), tf.bool)

                d = tfd.Masked(y_pred, mask)
                # sometimes event shape is [1] sometimes it's []
                if d.event_shape == 1:
                    marker_true = marker_true[:, None]
                nll = - d.log_prob(marker_true)
                marker_loss = tf.reduce_sum(nll)
                return marker_loss



        losses[markers[loss_index]] = loss  
        
    return losses


class Sigma(tfkl.Layer):
    """
    A layer that applies a softplus activation to its input to get a trainable positive vector
    """
    def __init__(self, output_dim, **kwargs):
       self.output_dim = [output_dim]
       super(Sigma, self).__init__(**kwargs)

    def build(self, input_shapes):
       self.kernel = self.add_weight(name='kernel', shape=self.output_dim, initializer='normal', trainable=True)
       super(Sigma, self).build(input_shapes)  

    def call(self, inputs):
       return tf.math.softplus(self.kernel)

    def compute_output_shape(self):
       return self.output_dim