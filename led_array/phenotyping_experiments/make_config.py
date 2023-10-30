import os
import yaml


TEMPLATE_PATH = "/home/hpinkard_waller/GitRepos/EncodingInformation/led_array/phenotyping_experiments/config_files/staging/template.yaml"

def make_config(changes):
    # load template and replace stuff as specified
    config = yaml.safe_load(open(TEMPLATE_PATH, "r"))

    # recursively update the config with the changes
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    config = update(config, changes)

    return config

# get the folder of the template
template_folder = os.path.dirname(TEMPLATE_PATH)
prefix = template_folder + os.sep + "Synthetic_Noise_"


# create a list of all the names of the yaml files
for photon_count in [50, 100, 200, 300]:
    for channel in ['DPC_Right', 'LED119', 'Brightfield']:
        for replicate in range(10):
            name = prefix + channel + "_" + str(photon_count) + f"_photons_replicate_{replicate}.yaml"

            # create yaml files with each name
            with open(name, "w") as f:    
                changes = {"data": {"channels": [channel],
                                    "synthetic_noise": {"photons_per_pixel": photon_count}},
                            "metadata": {"replicate": replicate}}

                contents = make_config(changes)

                # yaml.dump(contents, f)  
                # print to yaml file with nice formatting where each key is on a new line
                yaml.dump(contents, f, default_flow_style=False)     