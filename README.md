# Welcome to hSNN

This is the repository for [Temporal Hierarchy in Spiking Neural Networks]( https://arxiv.org/abs/2407.18838 ).

## Repo Content

- `requirements`: directory containing the requirements files for macOS and Linux
- `results`: directory where the results are stored

- The following folders will be generated at the first execution of the code:
  - `datasets/`: directory containing the audio spikes dataset [SHD](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/), [SSC](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) and [MNIST](https://yann.lecun.com/exdb/mnist/).
      - `audiospikes/`: directory containing the audio spikes dataset, for SHD and SSC

### Code description
- `utils_dataset.py`        : script containing the dataloaders for the datasets utilized in this work (SHD, SSC, MTS-XOR, (s)MNIST).
- `utils_initialization.py` : contains the hyper-parameters and the code to initialize the parameters (weights and time constants) in the hSNNs.
- `utils_normalization.py`  : contains the code for both layer_norm and batch_norm normalization.
- `models.py`               : includes the models for Leaky-Integrate-and-Fire neurons, as well as other helper functions
- `training.py`             : contains the training loop function, as well as the definition of the hSNN model
- `main.py`                 : the reference code to make a single run with given hyper-parameter, specified in command line (see below)
- `sweep.py`                : the code from which all the results are obtained, can be used to reproduce figures

### System Requirements (tested on)
- MacBook Pro (M2 Pro) with macOS Ventura 13.2.1
- Linux Ubuntu 22.04 LTS + cuda 11 (NVIDIA A6000)

We highly recommand using a GPU for training the model.

### Python Requirements
- Python 3.10.2
- We recommand creating a virtual environment: `python3 -m venv hsnn_venv`
#### macOS (Apple Silicon M2 Chip)
- requirements_macos.txt : `pip install -r requirements/requirements_macos.txt`
#### Linux (Ubuntu 20.04)
- requirements_linux.txt : `pip3 install -r requirements/requirements_gpu.txt`
- install jax: \
`pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- install PyTorch: \
`pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118`

## Running the Code

### Training
- run a single training: \
`python3 main.py --dataset_name $dataset_name --seed $seed --n_epochs $n_epochs`
- Default values for the parameters when running: `python3 main.py` 
(without arguments)
    - `$dataset_name`: 'shd'
    - `$seed`: 0
    - `$n_epochs`: 5

## Reproducing the Results

### Training
-- all the HPO and parameter sweeps to obtain figures from the main paper and supplementary material are included in the "sweep.py" script.

To run such script, use the following command:

`python3 sweep.py --sweep_name $sweep_name`

where you can choose between the following runs, by replacing $sweep_name with the following:

    - `hSNN_Figure_XX`: where XX indicates which figure you are interested in reproducing (for example, Figure 2c --> `hSNN_Figure_2c`). Note that figures in supplementary information feature an "S" in front: `hSNN_Figure_S1a`.

### Time Benchmarking (batch size = 256)
- macOS: 
  - SHD, 3 layers, 10  hidden neurons: ~0.5 s / epoch 
  - SHD, 4 layers, 32  hidden neurons: ~4 s / epoch 
  - SSC, 4 layers, 128 hidden neurons: ~40 s / epoch
- Linux: 
  - SHD, 4 layers, 32  hidden neurons: ~2.5 s / epoch 
  - SSC, 4 layers, 128 hidden neurons: ~30 s / epoch