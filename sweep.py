import os
import jax
import argparse
import numpy as np
from datetime import datetime

from utils_dataset import get_dataloader
from models import decoder_cum, decoder_sum, decoder_vlast, decoder_vmax
from utils_normalization import BatchNorm, LayerNorm
from utils_initialization import SimArgs, params_initializer
from training import train_hsnn_wandb

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".5" # needed because network is huge
os.environ["CUDA_VISIBLE_DEVICES"]="1"
jax.devices()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_name', type=str, default='hsnn_test')
    parsed = parser.parse_args()
    
    import wandb
    wandb.login()

    # Load the default parameters
    args = SimArgs()
    
    # setup the sweep
    sweep_config = {
        'method': 'grid',
    }

    metric = {
        'name': 'accuracy',
        'goal': 'maximize'
        }
    sweep_config['metric'] = metric

    config = {}
    for i, [key, value] in enumerate( zip( args.__dict__.keys(), args.__dict__.values()  ) ):
        config[key] = {'value': value}
    # update the parameters to sweep
        
    ### Delta_tau_train_alpha_False
    if parsed.sweep_name == 'Delta_tau_train_alpha_False':
        config['delta_tau'] = {'values':[0.09]} # [-0.075, -0.05, -0.025, -0.001, 0, 0.001, 0.025, 0.05, 0.075]
        config['seed'] = {'values':[3, 4]} # [0, 1, 2]
        config['n_epochs'] = {'value':40}
        config['n_layers'] = {'value':4}
        config['train_alpha'] = {'value':False}
        config['hierarchy_tau'] = {'value':True}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="Delta_tau_train_alpha_False")

    ### Tau_mem_train_alpha_False
    elif parsed.sweep_name == 'Tau_mem_train_alpha_False':
        config['tau_mem'] = {'values':[0.01, 0.05, 0.1, 0.2, 0.4, 0.8]}
        config['seed'] = {'values':[0,1,2,3,4]} # [0, 1, 2]
        config['n_epochs'] = {'value':40}
        config['n_layers'] = {'value':4}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="Tau_mem_train_alpha_False")

    ### Test
    else:
        config['seed'] = {'value':0}
        config['n_epochs'] = {'value':5}
        config['n_layers'] = {'value':4}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_test")

    ### Launch the sweep
    wandb.agent(sweep_id, train_hsnn_wandb)