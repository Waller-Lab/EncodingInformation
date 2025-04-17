


Specification of dependencies






Training code
Evaluation code
Pre-trained models
README file including table of results accompanied by precise commands to run/produce those results








## Validation of mutual information estimator (Section 3 in paper)

- Uses the `./led_array/estimator_and_microscopy_requirements.txt` environment



## Information estimation on imaging applications (Section 4 in paper)

### Fig 1
(Conceptual, no experiments)


### Fig 2
#### 2a
    TODO

#### 2b
    TODO

#### 2c
    TODO

#### 2d
- *Environment*: `led_microscopy_requirements.txt`
- *Data*: [BSCCM dataset](https://waller-lab.github.io/BSCCM/)
- *Experiments*
    - Records of trained models + hyperparameters in `./led_array/phenotyping_experiments/config_files`
    - The script for training models `./led_array/phenotyping_experiments/train_model.py`
- *Analysis/figure* using `./led_array/phenotyping_experiments/make_phenotyping_vs_mi_plot.ipynb`
    


### Fig 3
    TODO



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

#### S20a 
TODO

#### S20b
TODO

#### S20c 
TODO


### S21
- *Environment*: `estimator_and_microscopy_requirements.txt`
- *Experiments/figure* `mi_estimator_experiments/mi_of_background.ipynb`

### S22
TODO
