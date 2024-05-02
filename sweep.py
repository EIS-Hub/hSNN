import os
import jax
import argparse
import numpy as np
from datetime import datetime

from utils_dataset import get_dataloader
from models import decoder_cum, decoder_sum, decoder_vlast, decoder_vmax
from utils_normalization import BatchNorm, LayerNorm
from utils_initialization import SimArgs, params_initializer
from training import train_hsnn

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".75" # needed because network is huge
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

    ### Hierarchy shape
    if parsed.sweep_name == 'Hierarchy_shape_SHD':
        print('Starting with the sweep on Hierarchy function')
        config['tanh_coef'] = {'values':[0.1, 0.25, 0.5, 1.]}
        config['tanh_center'] = {'values':[0, 0.25, 0.5, 0.75, 1]}
        config['seed'] = {'values':[0, 1, 2, 3, 4]} # [0, 1, 2, 3, 4] [5,6,7,8,9]
        config['tau_mem'] = {'value':0.1}
        config['hierarchy_tau'] = {'value':'tanh'}
        config['delta_tau'] = {'value':0.15}
        config['n_epochs'] = {'value':60}
        config['n_layers'] = {'value':6}
        config['n_hid'] = {'value':32}
        config['experiment_name'] = {'value':parsed.sweep_name} # {'value':'test_tau_mem'} #
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)
        
    ### Delta_tau_train_alpha_False
    elif parsed.sweep_name == 'Delta_tau_train_alpha_False':
        print('Starting with the sweep on Delta Tau (hierarchy)')
        config['delta_tau'] = {'values':[-0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075, 0.09]} # [-0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075]
        config['seed'] = {'values':[3,4]} # [0, 1, 2, 3, 4] [5,6,7,8,9]
        config['n_epochs'] = {'value':60}
        config['n_layers'] = {'value':3}
        config['train_alpha'] = {'value':False}
        config['distrib_tau'] = {'value':'unif'}
        config['hierarchy_tau'] = {'value':True}
        config['recurrent'] = {'value':False} ### ----> be careful here!
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)

    ### Tau_mem_train_alpha_False
    elif parsed.sweep_name == 'Tau_mem_train_alpha_False':
        print('Starting with the sweep on the Time Constant')
        config['tau_mem'] = {'values':[0.01, 0.05, 0.1, 0.2, 0.4]} # [0.01, 0.05, 0.1, 0.2, 0.4]
        config['seed'] = {'values':[0,1,2]} # [0, 1, 2]
        config['distrib_tau'] = {'values':[True, False]}
        # config['recurrent'] = {'value':True} ### ----> be careful here!
        config['n_epochs'] = {'value':60}
        config['n_layers'] = {'value':4}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)

    ### Num_Layers
    elif parsed.sweep_name == 'Num_layers_train_alpha_False':
        print('Starting with the sweep on the Time Constant')
        config['n_layers'] = {'values':[3,4,5]}
        config['n_hid'] = {'values':[256]}
        config['seed'] = {'values':[0,1,2]} # [0, 1, 2]
        config['hierarchy_tau'] = {'values':[True, False]}
        config['n_epochs'] = {'value':60}
        config['delta_tau'] = {'value':0.075}
        config['distrib_tau'] = {'value':'uniform'}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)

    ### MTS_XOR: tau_mem
    elif parsed.sweep_name == 'MTS_XOR_tau_mem':
        print('Starting with the sweep on the Time Constant')
        config['n_layers'] = {'values':[3,4]}
        config['seed'] = {'values':[0,1,2,3,4]}
        config['tau_mem'] = {'values':[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.6]}
        config['dataset_name'] = {'value':'mts_xor'} # Task name!
        config['n_hid'] = {'values':[10]}
        config['hierarchy_tau'] = {'value':False}
        config['n_epochs'] = {'value':60}
        config['distrib_tau'] = {'value':'normal'}
        config['n_in'] = {'value':40}; config['n_out'] = {'value':2}
        config['decoder'] = {'value':'vmem_time'}; config['time_max'] = {'value':1.0}
        config['timestep'] = {'value':1./100}; config['time_max'] = {'value':1.0}
        config['tau_out'] = {'value':0.05}; config['distrib_tau_sd'] = {'value':0.1}
        config['batch_size'] = {'value':512}
        config['experiment_name'] = {'value':parsed.sweep_name}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)

    ### MTS_XOR: delta_tau
    elif parsed.sweep_name == 'MTS_XOR_delta_tau':
        print('Starting with the sweep on the Delta_Tau (Hierarchy)')
        config['n_layers'] = {'values':[3,4]}
        config['seed'] = {'values':[5,6,7,8,9]} # 0,1,2,3,4
        config['delta_tau'] = {'values':[-0.5, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.5]}
        config['dataset_name'] = {'value':'mts_xor'} # Task name!
        config['n_hid'] = {'value':10}
        config['tau_mem'] = {'value':0.3}
        config['hierarchy_tau'] = {'value':'tanh'}
        config['n_epochs'] = {'value':60}
        config['distrib_tau'] = {'value':'normal'}
        config['n_in'] = {'value':40}; config['n_out'] = {'value':2}
        config['decoder'] = {'value':'vmem_time'}; config['time_max'] = {'value':1.0}
        config['timestep'] = {'value':1./100}; config['time_max'] = {'value':1.0}
        config['tau_out'] = {'value':0.05}; config['distrib_tau_sd'] = {'value':0.1}
        config['batch_size'] = {'value':512}
        config['experiment_name'] = {'value':parsed.sweep_name}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)

    ### Test
    else:
        print('Invalid sweep name')
        config['seed'] = {'values':[0]}
        config['n_epochs'] = {'value':2}
        config['n_layers'] = {'value':3}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_test")

    ### Launch the sweep
    wandb.agent(sweep_id, train_hsnn)