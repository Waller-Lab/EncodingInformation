# Radio Astronomy for Black Hole Imaging

Mutual information estimates for the black hole imaging application use the code and environment as described in the main repo.
For synthetic black hole generation, measurement generation, and reconstructions a separate enironment is used. This environment relies on two main packages:
- `ehtim` https://achael.github.io/eht-imaging/ 
- `pynoisy` https://github.com/aviadlevis/pynoisy

A copy of the environment is included in `blackhole.yml`. To set up this environment you can create it from the file by running `conda env create -f blackhole.yml`, which will create an environment named `blackhole`. 

Note that `pynoisy` has some additional dependencies that may require special installation steps, please see https://github.com/aviadlevis/pynoisy for detailed instructions and troubleshooting.