import os
import argparse
import yaml
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import wandb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import numpy as np


from utils import *

cmap = ListedColormap(['red', 'green', 'blue', 'white'])

def main(config_path, gpu_idx):
    # Set up GPU
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    # Load config
    with open(config_path, 'r') as file:
        h_params = yaml.safe_load(file)

    # Initialize the model
    model_key = jax.random.PRNGKey(h_params['model_key'])
    P, F, K = h_params['P'], h_params['F'], h_params['K']
    sensor_shape = (h_params['mask_size'], h_params['mask_size'], h_params['num_channels'])
    gamma = h_params['gamma']
    model = FullModel(sensor_shape, P, F, K, gamma, model_key)

    if h_params['mask_path']:
        mask = np.load(h_params['mask_path'])
        model = eqx.tree_at(lambda m: m.sensor_layer, model, replace=model.sensor_layer.update_w(mask))

    try:
        # deserialize the model
        model = eqx.tree_deserialise_leaves(h_params['model_path'], model)
    except:
        pass

    # Initialize the optimizers
    def param_labels(listed_model):
        model = listed_model[0]
        labels = jax.tree.map(lambda _: 'sensor_layer', model)
        labels = eqx.tree_at(lambda m: m.reconstruction_model, labels, 'reconstruction_model')
        return [labels]

    optimizer = optax.multi_transform(
        {'sensor_layer': optax.adamw(learning_rate=h_params["mask_lr"], b1=h_params["b1"], b2=h_params["b2"]),
        'reconstruction_model': optax.adamw(learning_rate=h_params["lr"], b1=h_params["b1"], b2=h_params["b2"])},
        param_labels=param_labels
    )

    opt_state = optimizer.init(eqx.filter([model], eqx.is_array))

    # Instantiate the data loader
    batch_size = h_params['n_batch']
    patch_size = (F, F)
    data_path = h_params['data_path']
    total_patches = h_params['total_patches']
    data_key = jax.random.PRNGKey(h_params['data_key'])

    # Create train and test loaders separately
    train_loader, train_indices = create_patch_loader(
        zarr_path = os.path.join(data_path, 'train'), 
        batch_size=batch_size, 
        patch_size=patch_size, 
        key=data_key, 
        total_patches=total_patches,
        num_workers=24
    )
    test_loader, test_indices = create_patch_loader(
        zarr_path = os.path.join(data_path, 'val'), 
        batch_size=batch_size, 
        patch_size=patch_size, 
        key=data_key, 
        total_patches=int(total_patches * h_params['val_size']),
        num_workers=24
    )

    # Iterate through the train loader to get the mean of the last channel over the entire dataset
    if h_params['data_mean_path'] is None:
        train_loader_iter = iter(train_loader)
        cum_mean = 0
        count = 0
        for images, _ in tqdm(train_loader_iter):
            cum_mean += jnp.mean(images[:,:,:,3])
            count += 1
        mean_last_channel = cum_mean / count
    else:
        mean_last_channel = np.load(h_params['data_mean_path'])
    scale_factor = h_params['mean_photons'] / mean_last_channel
    h_params['scale_factor'] = scale_factor

    if h_params['e2e'] == True:

        @eqx.filter_value_and_grad
        def loss(model, x, y, alpha):
            pred_y = jax.vmap(model)(x, alpha)
            return jnp.mean((y - pred_y) ** 2)
        @eqx.filter_jit
        def step_model(model: eqx.Module,
                    optimizer: optax.GradientTransformation,
                    opt_state: optax.OptState,
                    images: jnp.ndarray,
                    targets: jnp.ndarray,
                    alpha:jnp.ndarray
                    ):
            # Compute the gradients with respect to params (static parts are not differentiated)
            loss_val, grads = loss(model, images, targets, alpha)
            # Update parameters
            updates, new_opt_state = optimizer.update([grads], opt_state, [model])
            updates = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), updates)
            model = eqx.apply_updates(model, updates[0])
            return model, new_opt_state, loss_val
        
    else:
        filter_spec = create_filter_spec(model, [model.sensor_layer])
        @eqx.filter_value_and_grad
        def loss(diff_model, static_model, x, y, alpha):
            model = eqx.combine(diff_model, static_model)
            pred_y = jax.vmap(model)(x, alpha)
            return jnp.mean((y - pred_y) ** 2)
        
        @eqx.filter_jit
        def step_model(model, optimizer, opt_state, images, targets, alpha):
            diff_model, static_model = eqx.partition(model, filter_spec)
            loss_val, grads = loss(diff_model, static_model, images, targets, alpha)
            updates, new_opt_state = optimizer.update([grads], opt_state, [model])
            updates = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), updates)
            model = eqx.apply_updates(model, updates[0])
            return model, new_opt_state, loss_val


    # Training function
    def train(
            model: eqx.Module,
            optimizer: optax.GradientTransformation,
            opt_state: optax.OptState,
            train_loader: PredeterminedPatchLoader,
            test_loader: PredeterminedPatchLoader,
            h_params: dict
    ):
        name = h_params['model_name']
        num_steps = h_params['num_steps']
        save_freq = h_params['save_freq']
        disp_freq = h_params['disp_freq']
        val_freq = h_params['val_freq']
        gamma = h_params['gamma']
        scale_factor = h_params['scale_factor']
        
        # define noise key
        noise_key = jax.random.PRNGKey(0)

        # Initialize wandb
        wandb.init(project='MI_CFA', config=h_params, name=name)
        wandb_log = {}

        # Initialize the validation patches
        disp_images, disp_targets = next(iter(test_loader))
        disp_images = disp_images
        # scale the images and add noise
        disp_images = disp_images*scale_factor
        # add gaussian approximation of poisson noise
        noise_key, subkey = jax.random.split(noise_key)
        disp_images = disp_images + jax.random.normal(subkey, shape=disp_images.shape) * jnp.sqrt(disp_images)
        # divide by scale factor
        disp_images = disp_images / scale_factor
        # clip the images
        disp_images = disp_images.clip(0, 1)


        disp_targets = disp_targets * 3

        # Log the validation targets
        fig, axs = plt.subplots(5, 5, figsize=(15, 15))
        for i in range(25):
            ax = axs[i // 5, i % 5]
            ax.imshow((disp_targets[i]**(1/2.2)).clip(0, 1))
            ax.axis("off")
        plt.tight_layout()
        wandb_log["val_targets"] = wandb.Image(fig)
        plt.close()


        train_loader_iter = iter(train_loader)
        for step in tqdm(range(num_steps)):
            if step != 0:
                wandb_log = {}
            
            try:
                images, targets = next(train_loader_iter)
            except StopIteration:
                train_loader.shuffle_indices()
                train_loader_iter = iter(train_loader)
                images, targets = next(train_loader_iter)
            
            # scale the images and add noise
            images = images*scale_factor
            # add gaussian approximation of poisson noise
            noise_key, subkey = jax.random.split(noise_key)
            images = images + jax.random.normal(subkey, shape=images.shape) * jnp.sqrt(images)
            # divide by scale factor
            images = images / scale_factor
            # clip the images
            images = images.clip(0, 1)

            # make alpha an array of size n_batch
            if h_params['e2e'] == True:
                alpha_val = 1 + (gamma * step) ** 2
                alpha = jnp.full((images.shape[0], 1), alpha_val)
            else:
                alpha_val = 10000
                alpha = jnp.full((images.shape[0], 1), alpha_val)
            model, opt_state, loss = step_model(model, optimizer, opt_state, images, targets, alpha)
            wandb_log['loss'] = loss

            if (step % disp_freq) == 0:
                alpha = jnp.full((disp_images.shape[0], 1), alpha_val)
                recons = jax.vmap(model)(disp_images, alpha)
                recons = (3*recons)**(1/2.2)
                recons = recons.clip(0, 1)
    
                fig, axs = plt.subplots(5, 5, figsize=(15, 15))
                for i in range(25):
                    ax = axs[i // 5, i % 5]
                    ax.imshow(recons[i])
                    ax.axis("off")
                plt.tight_layout()
                wandb_log["reconstructions"] = wandb.Image(fig)
                plt.close()

                # display the sensor
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                cax = ax.imshow(model.sensor_layer.w.argmax(axis=-1), cmap=cmap)
                ax.axis("off")
                wandb_log["sensor"] = wandb.Image(fig)
                plt.close()

                # display how binary the sensor is
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                cax = ax.imshow(jnp.max(jax.nn.softmax(model.sensor_layer.w * alpha_val, axis=-1), axis=-1))
                ax.axis("off")
                cbar = fig.colorbar(cax, ax=ax)
                cax.set_clim(0, 1) 
                wandb_log["sensor_binary"] = wandb.Image(fig)
                plt.close()
                
            if (step % val_freq) == 0:
                val_loss = 0
                val_count = 0
                for val_images, val_targets in tqdm(test_loader, desc="Validation"):
                    val_count += 1
                    # scale the images and add noise
                    val_images = val_images*scale_factor
                    # add gaussian approximation of poisson noise
                    noise_key, subkey = jax.random.split(noise_key)
                    val_images = val_images + jax.random.normal(subkey, shape=val_images.shape) * jnp.sqrt(val_images)
                    # divide by scale factor
                    val_images = val_images / scale_factor
                    # clip the images
                    val_images = val_images.clip(0, 1)
                    alpha = jnp.full((val_images.shape[0], 1), alpha_val)
                    recons = jax.vmap(model)(val_images, alpha)
                    val_loss += jnp.mean((recons - val_targets) ** 2)
                val_loss /= val_count
                print(f"Validation loss: {val_loss}")
                wandb_log['val_loss'] = val_loss

            if (step % save_freq) == 0:
                try:
                    os.makedirs(os.path.join("models", name))
                except:
                    pass
                eqx.tree_serialise_leaves(
                    os.path.join("models", name, f"step_{str(step).zfill(7)}.eqx"),
                    model
                )

            wandb.log(wandb_log)
                
        return model, opt_state

    # Run training
    trained_model, final_opt_state = train(model, optimizer, opt_state, train_loader, test_loader, h_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reconstruction training")
    parser.add_argument("config_path", type=str, help="Path to the YAML config file")
    parser.add_argument("gpu_idx", type=int, help="GPU index to use")
    args = parser.parse_args()

    main(args.config_path, args.gpu_idx)