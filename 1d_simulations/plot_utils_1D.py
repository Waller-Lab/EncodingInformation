from signal_utils_1D import *
from cleanplots import *
from signal_utils_1D import *
import cmcrameri # required in order to register the colormaps with Matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize



def plot_in_spatial_coordinates(ax, signal, label=None, show_upsampled=True, show_samples=False, 
                                color_samples=False, vertical_line_indices=None, full_height_vertical_lines=False,
                                sample_point_indices=None, horizontal_line_indices=None, 
                                upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH,
                                 markersize=8, marker='o', random_colors=False, center=False, plot_lim=1, color=None, 
                                 colors=None, erasure_mask=None, xlabel='Space', ylabel='Intensity', **kwargs):


    if signal.shape[-1] == UPSAMPLED_SIGNAL_LENGTH:
        num_nyquist_samples = NUM_NYQUIST_SAMPLES
    else:
        num_nyquist_samples = signal.shape[-1]                             

    def plot_one_signal(signal, sample_point_indices=None, color=None, erasure_mask=erasure_mask):
        if signal.size != UPSAMPLED_SIGNAL_LENGTH:
            x, x_upsampled, upsampled_signal = upsample_signal(signal, num_nyquist_samples=signal.size,
                                                            upsampled_signal_length=upsampled_signal_length, return_domain=True)
        else:
            upsampled_signal = signal
            x_upsampled = np.linspace(0, 1, upsampled_signal.shape[-1], endpoint=False)
            x = np.linspace(0, 1, num_nyquist_samples, endpoint=False)
        if show_upsampled:
            if center:
                upsampled_signal = np.roll(upsampled_signal, upsampled_signal.size // 2 - np.argmax(upsampled_signal))

            if erasure_mask is not None:
                erasure_mask = np.repeat(erasure_mask, UPSAMPLED_SIGNAL_LENGTH // NUM_NYQUIST_SAMPLES)
                upsampled_signal *= erasure_mask.astype(float)
        

            ax.plot(x_upsampled, upsampled_signal, label=label, linewidth=2.1, color=color, 
                    # zorder=3, 
                    **kwargs)
            # get the color used for the line
            color = ax.get_lines()[-1].get_color()
        if show_samples:
            if sample_point_indices is None:
                sample_point_indices = np.arange(num_nyquist_samples)
            sample_point_indices = np.array(sample_point_indices)
            ax.plot(x[sample_point_indices], signal[sample_point_indices], 
                                       marker,
                                        markersize=markersize, 
                                        # zorder=3,
                                        color='k' if not color_samples else color,
                                        label=None if show_upsampled else label, 
                                        **kwargs)
            

    if vertical_line_indices is not None:
        x = upsample_signal(signal, return_domain=True, num_nyquist_samples=num_nyquist_samples)[0]
        # find highest value over all signals at verical_line_indices
        if full_height_vertical_lines:
            max_values = np.ones(len(vertical_line_indices))
        else:
            max_values = np.max(signal.reshape(-1, num_nyquist_samples)[..., vertical_line_indices], axis=0)
        for i, max_val in zip(vertical_line_indices, max_values):
            # plot line going from 0 to max_val
            ax.plot([x[i], x[i]], [0, max_val], 'k--')
    if horizontal_line_indices is not None:
        x = upsample_signal(signal, return_domain=True, num_nyquist_samples=num_nyquist_samples)[0]
        # find highest value over all signals at verical_line_indices
        y_values = np.max(signal.reshape(-1, num_nyquist_samples)[..., vertical_line_indices], axis=0)
        for x_index, y in zip(horizontal_line_indices, y_values):
            # plot line going from 0 to max_val
            ax.plot([0, x[x_index]], [y, y], 'k--')
            

    if len(signal.shape) == 1:
        plot_one_signal(signal, sample_point_indices=sample_point_indices, color=color)
    else:
        for i in range(signal.shape[0]):
            color = colors[i] if colors is not None else None
            plot_one_signal(signal[i], sample_point_indices=sample_point_indices, 
                                color=color if not random_colors else onp.random.rand(3))
                
    clear_spines(ax)
    ax.set(ylabel=ylabel, xlim=[0, 1], xlabel=xlabel, ylim=[0, plot_lim], xticks=[0, 1], yticks=[0, plot_lim])




def plot_object(ax, signal, colors=None, **kwargs):
    for i, o in enumerate(signal.reshape(-1, signal.shape[-1])):
        ax.plot(np.linspace(0,1, o.size), o, **kwargs, color=colors[i] if colors is not None else None)
    ax.set(xlabel='Space', ylabel='Intensity', xlim=[0, 1], xticks=[0,1], ylim=[0, signal.max()], yticks=[0, signal.max()])
    sparse_ticks(ax)

    clear_spines(ax)
    

def plot_intensity_coord_histogram(ax, signals, sample_point_indices=(3, 4), **kwargs):
    bins = np.linspace(0, 1, 50)  
    h = ax.hist2d(signals[:, sample_point_indices[0]], signals[:, sample_point_indices[1]], 
              bins=[bins, bins], cmap='inferno', density=True)
    ax.set(xlabel='$I_1$', ylabel='$I_2$')
    # dash white 45 degree line
    ax.plot([0, 1], [1, 0], 'w--')
    # show colorbar with not ticks
    plt.colorbar(h[3], ax=ax, ticks=[])
    # make an axis label for the colorbar
    ax.text(1.2, 0.5, 'Probability', rotation=90, va='center', ha='left', transform=ax.transAxes)

    

def plot_in_intensity_coordinates(ax, signal, markersize=30, random_colors=False,
                                color=None, differentiate_colors=False, sample_point_indices=(3,4), plot_lim=1,
                                **kwargs):
    # plot the line y = -x + 
    # only plot this if there's nothing in the axes already
    if len(ax.lines) == 0:
        ax.plot([0, 1], [1, 0], 'k--', zorder=-1)
        ax.set(xlim=[0, 1], ylim=[0, 1], xlabel='$I_1$', ylabel='$I_2$')
    if differentiate_colors:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if len(signal.shape) == 1:
            color = color[0]
        else: 
            color = [color[i % len(color)] for i in range(signal.shape[0])]
    if random_colors:
        # generate a list of random colors of length signal.shape[0]
        color = onp.random.rand(signal.shape[0], 3)

    ax.scatter(signal[..., sample_point_indices[0]], signal[..., sample_point_indices[1]], s=markersize, 
               c=color, zorder=3,
               **kwargs)
               
            #    size=markersize, color=color)
    ax.set_aspect('equal')
    ax.set(xlim=[0, plot_lim], ylim=[0, plot_lim], xticks=[0, plot_lim], yticks=[0, plot_lim])

    clear_spines(ax)
    # plot again with x as marker

def make_PSF_output_signal_plot(ax, params, objects, erasure_mask, noise_sigma,
                                    mi, N_signals_to_plot=5, sampling_indices=(3, 4)):
  
  # plot the initial and optimized convolutional encoders

  plot_in_spatial_coordinates(ax[0], param_vec_to_signal(params), show_samples=False, center=True)
  ax[0].set_title('PSF')


  def make_noisy_output_signals(params, objects, erasure_mask, noise_sigma):
      return add_gaussian_noise_numpy(upsample_signal(conv_forward_model_with_erasure(params, objects, erasure_mask)), noise_sigma)

  # plot the output signals
  plot_in_spatial_coordinates(ax[1], make_noisy_output_signals(params, objects[:N_signals_to_plot],
                                                                   erasure_mask, noise_sigma), show_samples=False)

  # plot the output signals in intensity coordinates
  plot_in_intensity_coordinates(ax[2], conv_forward_model_with_erasure(params, objects, erasure_mask),
                                markersize=5, differentiate_colors=True, sample_point_indices=sampling_indices)


def make_delta_fn_object_PSF_plot(ax, params, objects, erasure_mask, noise_sigma, intensity_lim_max=1, psf_y_max=1,
                                  N_signals_to_plot=5, N_objects_to_plot=100,
                                sampling_indices=(3, 4), titles=False):
    


    ### plot delta function objects, colored by position
    colormap = plt.get_cmap('cmc.romaO')
    object_colors = [colormap((np.argmax(o) + 0.5) / o.size) for o in objects]
    example_object_colors = [colormap((np.argmax(o) + 0.5) / o.size) for o in objects]
    
    plot_object(ax[0], objects[:N_objects_to_plot], colors=example_object_colors[:N_objects_to_plot])
    if titles:
        ax[0].set_title('Object')
    

    # show a color bar on the x axis
    x_positions = np.linspace(0, 1, 100)
    colors = colormap(x_positions) 
    norm = Normalize(vmin=min(x_positions), vmax=max(x_positions))
    scalar_mappable = ScalarMappable(norm=norm, cmap=colormap)
    fig = ax[0].get_figure()
    cbar = fig.colorbar(scalar_mappable, ax=ax[0], orientation='horizontal', pad=0.02)
    # set the tick labels of the colorbar
    cbar.set_ticks([0, 1])

    # center the kernel
    kernel = conv_kernel_from_params(params)
    kernel_display = np.roll(kernel, kernel.size // 2 - np.argmax(kernel))
    kernel_for_conv = np.fft.fftshift(kernel_display)
    params_for_conv =  np.concatenate(real_imag_params_from_signal(kernel_for_conv))
  
    # compute signals with optimized kernels
    signals = conv_forward_model_with_erasure(params_for_conv, objects, erasure_mask, align_center=False)

    plot_in_spatial_coordinates(ax[1], kernel_display, show_samples=False, color='k')
    if titles:
        ax[1].set_title('PSF')
    ax[1].set(yticks=[0, intensity_lim_max], ylim=[0, psf_y_max], xticks=[0,1], xticklabels=[-0.5, 0.5])


    plot_in_spatial_coordinates(ax[2], signals[:N_signals_to_plot], show_samples=False, 
                                vertical_line_indices=sampling_indices, full_height_vertical_lines=True,
                                colors=example_object_colors, erasure_mask=erasure_mask)
    
    ax[2].set(ylabel=None, yticks=[0, intensity_lim_max], yticklabels=[], ylim=[0, intensity_lim_max], xticks=[0, 1])


    def make_noisy_output_signals(params, objects, erasure_mask, noise_sigma):
        return add_gaussian_noise_numpy(upsample_signal(conv_forward_model_with_erasure(params, objects, erasure_mask)), noise_sigma)

    # plot all output signals in intensity coordinates
    # signals = conv_forward_model_with_erasure(params, objects, erasure_mask, align_center=False)

    plot_in_intensity_coordinates(ax[3], make_noisy_output_signals(params_for_conv, objects, erasure_mask, noise_sigma),
                                   markersize=5, color=object_colors,
                                sample_point_indices=sampling_indices, plot_lim=intensity_lim_max)





