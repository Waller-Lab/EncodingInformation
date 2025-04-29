

(Conceptual, no experiments)


### Fig 2
#### 2a
- *Environment*: `lcfa_requirements.txt`
- *Data*: [Shi's Re-processing of Gehler's Raw Dataset](https://www.cs.sfu.ca/~colour/data/shi_gehler/)
- *Experiments*: 
    - Script for E2E optimzation `./color_filter_array/recon.py` with config files `./color_filter_array/e2e_configs`
    - Script for mutual information verfication `./color_filter_array/mi_calculation.py` with config files `./color_filter_array/mi_configs`
    - Script for reconstruction training `./color_filter_array/recon.py` with config files `./color_filter_array/recon_configs`
- *Analysis/figure* using  `./color_filter_array/recon_validation.py`, `./color_filter_array/results_viewer.ipynb`, and `./color_filter_array/sandbox.ipynb`

#### 2b
- *Environment*: `astronomy_requirements.txt` and `lensless_requirements.txt`
- *Data*: `./radio_astronomy/2024_06_07_generate_many_black_holes.py` and `./radio_astronomy/2024_10_09_generate_black_hole_measurements_updated_template.py`
- *Experiments*: 
    - The script for mutual information estimation `./radio_astronomy/2024_10_11_mi_estimation_template.py`
    - The script for black hole image reconstruction `./radio_astronomy/2024_10_09_reconstruction_template.py`
- *Analysis/figure* using `./radio_astronomy/2024_10_15_mi_vs_reconstruction.ipynb` and `./radio_astronomy/2024_10_09_make_black_hole_figure.ipynb`

#### 2c
- *Environment*: `lensless_requirements.txt`
- *Data*: [CIFAR10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- *Experiments*:
    - The script for mutual information estimation `./lensless_imager/2024_10_23_pixelcnn_cifar10_updated_api_reruns_smaller_lr.py`
    - The script for image reconstruction `./lensless_imager/2024_10_22_sweep_unsupervised_wiener_deconvolution_per_lens.py`
- *Analysis/figure* using `./lensless_imager/2024_10_23_mi_vs_deconvolution_plots_cifar10_figure.ipynb`

#### 2d
- *Environment*: `led_microscopy_requirements.txt`
- *Data*: [BSCCM dataset](https://waller-lab.github.io/BSCCM/)
- *Experiments*
    - Records of trained models + hyperparameters in `./led_array/phenotyping_experiments/config_files`
    - The script for training models `./led_array/phenotyping_experiments/train_model.py`
- *Analysis/figure* using `./led_array/phenotyping_experiments/make_phenotyping_vs_mi_plot.ipynb`
    


### Fig 3
- *Environment*: `lcfa_requirements.txt`
- *Data*: [Shi's Re-processing of Gehler's Raw Dataset](https://www.cs.sfu.ca/~colour/data/shi_gehler/)
- *Experiments*: 
    - Script for IDEAL optimzation `./color_filter_array/ideal_optimization.py` with config files `./color_filter_array/ideal_configs`
    - Script for mutual information verfication `./color_filter_array/mi_calculation.py` with config files `./color_filter_array/mi_configs`
    - Script for reconstruction training `./color_filter_array/recon.py` with config files `./color_filter_array/recon_configs`
- *Analysis/figure* using  `./color_filter_array/recon_validation.py`, `./color_filter_array/results_viewer.ipynb`, and `./color_filter_array/sandbox.ipynb`

## Supplementary figures

### S2
Conceptual, no experiments

### S3
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `simulations_1d/6_rayleigh_2_point_1_point.ipynb`

### S4
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `simulations_1d/2_object_dependent_encoder_range.ipynb` and `4_conceptual_signal_space_figure.ipynb`

### S5
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `simulations_1d/5_mi_vs_bandlimited_signal_space.ipynb`

### S6
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `simulations_1d/5_mi_vs_bandlimited_signal_space.ipynb`

### S7
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `simulations_1d/mi_vs_signal_bandwidth.ipynb`, `simulations_1d/mi_vs_snr.ipynb`, `simulations_1d/mi_vs_sampling_density_SNRs.ipynb`


### S10
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `mi_estimator_experiments/stationary_GP_with_and_without_optimization.ipynb`

### S11
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `mi_estimator_experiments/stationary_GP_with_and_without_optimization.ipynb`, `mi_estimator_experiments/stationary_vs_full_covariance_gaussian.ipynb`

### S12
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `mi_estimator_experiments/patch_size_and_model_fits.ipynb`

### S13
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `mi_estimator_experiments/conceptual_MI_estimator_demonstration.ipynb`

### S14
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `mi_estimator_experiments/conceptual_stationary_process_samples.ipynb`

### S15
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `mi_estimator_experiments/gaussian_vs_pixelcnn_samples_lensless_imaging_01_04_2024.ipynb`

### S16
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `mi_estimator_experiments/estimator_consistency/MI_estimator_consistency.ipynb`

### S17
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `mi_estimator_experiments/estimator_consistency/MI_estimator_consistency.ipynb`

### S18
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `mi_estimator_experiments/estimator_consistency/conditional_from_noisy_samples.ipynb`, `mi_estimator_experiments/estimator_consistency/conditional_entropy_estimator_consistency.ipynb`

### S19
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `mi_estimator_experiments/analytic_gaussian_entropy_vs_test_set_nll.ipynb`



### S20

#### top row
- *Environment*: `lcfa_requirements.txt`
- *Data*: [Shi's Re-processing of Gehler's Raw Dataset](https://www.cs.sfu.ca/~colour/data/shi_gehler/)
- *Experiments*: 
    - Script for E2E optimzation `./color_filter_array/recon.py` with config files `./color_filter_array/e2e_configs`
    - Script for mutual information verfication `./color_filter_array/mi_calculation.py` with config files `./color_filter_array/mi_configs`
    - Script for reconstruction training `./color_filter_array/recon.py` with config files `./color_filter_array/recon_configs`
- *Analysis/figure* using  `./color_filter_array/recon_validation.py`, `./color_filter_array/results_viewer.ipynb`, and `./color_filter_array/sandbox.ipynb`

#### middle row
- *Environment*: `astronomy_requirements.txt` and `lensless_requirements.txt`
- *Data*: `./radio_astronomy/2024_06_07_generate_many_black_holes.py` and `./radio_astronomy/2024_10_09_generate_black_hole_measurements_updated_template.py`
- *Experiments*: 
    - The script for mutual information estimation `./radio_astronomy/2024_10_11_mi_estimation_template.py`
    - The script for black hole image reconstruction `./radio_astronomy/2024_10_09_reconstruction_template.py`
- *Analysis/figure* using `./radio_astronomy/2024_10_15_mi_vs_reconstruction.ipynb`

#### bottom row
- *Environment*: `lensless_requirements.txt`
- *Data*: [CIFAR10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- *Experiments*:
    - The script for mutual information estimation `./lensless_imager/2024_10_23_pixelcnn_cifar10_updated_api_reruns_smaller_lr.py`
    - The script for image reconstruction `./lensless_imager/2024_10_22_sweep_unsupervised_wiener_deconvolution_per_lens.py`
- *Analysis/figure* using `./lensless_imager/2024_10_23_mi_vs_deconvolution_plots_cifar10_figure.ipynb`


### S21
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `mi_estimator_experiments/mi_of_background.ipynb`

### S22
- *Environment*: `lensless_classifier_requirements.txt`
- *Data*: [CIFAR10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- *Experiments*:
    - The script for mutual information estimation `./lensless_imager/old_api/01_04_2024_pixelcnn_cifar10_10px_bias.py`
    - The script for image classification `./lensless_imager/11_14_2023_run_classifier_cifar10.py`
    - Classification models `./lensless_imager/old_api/classifier_results`
- *Analysis/figure* using `./lensless_imager/01_09_2024_mi_vs_classification_plots_updated_mi_cifar10.ipynb`


