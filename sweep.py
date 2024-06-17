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
os.environ["CUDA_VISIBLE_DEVICES"]="2"
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


    ###############################
    ############# SHD #############
    ###############################

    ### Hierarchy shape - Figure 2e
    if parsed.sweep_name == 'Hierarchy_shape_SHD':
        print('Starting with the sweep on Hierarchy function')
        config['tanh_coef'] = {'values':[0.1, 0.25, 0.5, 1.]}
        config['tanh_center'] = {'values':[0, 0.25, 0.5, 0.75, 1]}
        config['seed'] = {'values':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
        config['tau_mem'] = {'value':0.1}
        config['hierarchy_tau'] = {'value':'tanh'}
        config['delta_tau'] = {'value':0.15}
        config['n_epochs'] = {'value':60}
        config['n_layers'] = {'value':6}
        config['n_hid'] = {'value':32}
        config['experiment_name'] = {'value':parsed.sweep_name} # {'value':'test_tau_mem'} #
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)
        
    ### SHD Delta Tau Hiararchy - Figure 2 supplementary
    elif parsed.sweep_name == 'SHD_Delta_tau':
        print('Starting with the sweep on Delta Tau (hierarchy)')
        config['delta_tau'] = {'values':[0.1, 0.15, 0.18]} #{'values':[-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.18]}
        config['seed'] = {'values':[0, 1, 2, 3, 4] } # [0, 1, 2, 3, 4] [5,6,7,8,9]
        config['n_epochs'] = {'value':60}
        config['n_layers'] = {'value':6}
        config['tau_mem'] = {'value':0.1}
        config['n_hid'] = {'value':32}
        config['train_alpha'] = {'value':False}
        config['hierarchy_tau'] = {'value':'tanh'}
        config['recurrent'] = {'value':False} ### ----> be careful here!
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)

    ### SHD tau mem tuning - supplementary (Tau_mem_train_alpha_False)
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

    ### SHD number of layers with and without Hierarchy - Figure 2f
    elif parsed.sweep_name == 'SHD_Num_layers':
        print('Starting with the sweep on the Time Constant')
        config['n_layers'] = {'values':[3,4,5,6]}
        config['n_hid'] = {'values':[32,64,128]}
        config['seed'] = {'values':[0,1,2,3,4,5]} # [0, 1, 2]
        config['hierarchy_tau'] = {'value':'tanh'}
        config['n_epochs'] = {'value':60}
        config['delta_tau'] = {'values':[0, 0.15]}
        config['distrib_tau'] = {'value':'uniform'}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)

    ###############################
    ########### MTS-XOR ###########
    ###############################

    ### MTS_XOR: tau_mem - Figure 2b
    elif parsed.sweep_name == 'MTS_XOR_tau_mem':
        print('Starting with the sweep on the Time Constant')
        config['n_layers']      = {'values':[3,4]}
        config['seed']          = {'values':[0,1,2,3,4]}
        config['tau_mem']       = {'values':[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.6]}
        config['dataset_name']  = {'value':'mts_xor'} # Task name!
        config['n_hid']         = {'values':[10]}
        config['hierarchy_tau'] = {'value':False}
        config['n_epochs']      = {'value':60}
        config['distrib_tau']   = {'value':'normal'}
        config['n_in'] = {'value':40}; config['n_out'] = {'value':2}
        config['decoder'] = {'value':'vmem_time'}; config['time_max'] = {'value':1.0}
        config['timestep'] = {'value':1./100}; config['time_max'] = {'value':1.0}
        config['tau_out'] = {'value':0.05}; config['distrib_tau_sd'] = {'value':0.1}
        config['batch_size']    = {'value':512}
        config['experiment_name'] = {'value':parsed.sweep_name}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)

    ### MTS_XOR: delta_tau - Figure 2c
    elif parsed.sweep_name == 'MTS_XOR_delta_tau':
        print('Starting with the sweep on the Delta_Tau (Hierarchy)')
        config['n_layers']      = {'values':[3,4]}
        config['seed']          = {'values':[0,1,2,3,4,5,6,7,8,9]}
        config['delta_tau']     = {'values':[-0.5, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.5]}
        config['dataset_name']  = {'value':'mts_xor'} # Task name!
        config['n_hid']         = {'value':10}
        config['tau_mem']       = {'value':0.3}
        config['hierarchy_tau'] = {'value':'tanh'}
        config['n_epochs']      = {'value':60}
        config['distrib_tau']   = {'value':'normal'}
        config['n_in'] = {'value':40}; config['n_out'] = {'value':2}
        config['decoder'] = {'value':'vmem_time'}; config['time_max'] = {'value':1.0}
        config['timestep'] = {'value':1./100}; config['time_max'] = {'value':1.0}
        config['tau_out'] = {'value':0.05}; config['distrib_tau_sd'] = {'value':0.1}
        config['batch_size'] = {'value':512}
        config['experiment_name'] = {'value':parsed.sweep_name}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)

    ###############################
    ###### Convolutional SNN ######
    ###############################

    ### SHD SCNNL: dilation tuning
    elif parsed.sweep_name == 'SHD_SCNN_dilation_tuning':
        print('SHD, SCNN, tuning of the dilation started')
        config['seed']          = {'values':[0,1,2,3,4]}
        config['conv_dilation'] = {'values':[3,5,7,12,24]}
        config['convolution']   = {'value':True}
        config['hierarchy_conv']= {'value':False}
        config['conv_kernel']   = {'value':5} # we initially fix the kernel size to 5
        config['freq_shift']    = {'value':10}
        config['delta_ker']     = {'value':0}
        config['n_layers']      = {'value':4}
        config['nb_steps']      = {'value':200}
        config['n_hid']         = {'value':128}
        config['dropout_rate']  = {'value':0.4}
        config['l2_lambda']     = {'value':1e-4}
        config['tau_mem']       = {'value':0.02}
        config['n_epochs']      = {'value':60}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)

    ### SHD SCNNL: kernel size tuning
    elif parsed.sweep_name == 'SHD_SCNN_kernel_tuning':
        print('SHD, SCNN, tuning of the kernel sizes started')
        config['seed']          = {'values':[0,1,2,3,4]}
        config['conv_kernel']   = {'values':[3,5,7,9]}
        config['convolution']   = {'value':True}
        config['hierarchy_conv']= {'value':False}
        config['conv_dilation']   = {'value':5}
        config['freq_shift']    = {'value':10}
        config['delta_ker']     = {'value':0}
        config['n_layers']      = {'value':4}
        config['nb_steps']      = {'value':200}
        config['n_hid']         = {'value':128}
        config['dropout_rate']  = {'value':0.4}
        config['l2_lambda']     = {'value':1e-4}
        config['tau_mem']       = {'value':0.02}
        config['n_epochs']      = {'value':60}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)

    ### SHD SCNNL: kernel Hierarchy
    elif parsed.sweep_name == 'SHD_SCNN_kernel_Hierarchy':
        print('SHD, SCNN, Hierarchy of kernel sizes')
        config['seed']          = {'values':[0,1,2,3,4]}
        config['delta_ker']     = {'values':[-4,-3,-2,-1,0,1,2,3,4]}
        config['convolution']   = {'value':True}
        config['hierarchy_conv']= {'value':'kernel'}
        config['conv_dilation'] = {'value':5}
        config['conv_kernel']   = {'value':5}
        config['freq_shift']    = {'value':10}
        config['n_layers']      = {'value':3}
        config['nb_steps']      = {'value':200}
        config['n_hid']         = {'value':128}
        config['dropout_rate']  = {'value':0.4}
        config['l2_lambda']     = {'value':1e-4}
        config['tau_mem']       = {'value':0.02}
        config['n_epochs']      = {'value':60}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)

    ### SHD SCNNL: dilation Hierarchy
    elif parsed.sweep_name == 'SHD_SCNN_dilation_Hierarchy':
        print('SHD, SCNN, Hierarchy of kernel sizes')
        config['seed']          = {'values':[0,1,2,3,4]}
        config['delta_dil']     = {'values':[-3,-2,-1,0,1,2,3]}
        config['convolution']   = {'value':True}
        config['hierarchy_conv']= {'value':'dilation'}
        config['conv_dilation'] = {'value':5}
        config['conv_kernel']   = {'value':5}
        config['delta_ker']     = {'value':0}
        config['freq_shift']    = {'value':10}
        config['n_layers']      = {'value':3}
        config['nb_steps']      = {'value':200}
        config['n_hid']         = {'value':128}
        config['dropout_rate']  = {'value':0.4}
        config['l2_lambda']     = {'value':1e-4}
        config['tau_mem']       = {'value':0.02}
        config['n_epochs']      = {'value':60}
        sweep_config['parameters'] = config
        sweep_id = wandb.sweep(sweep_config, project="hsnn_"+parsed.sweep_name)

    ### SHD SCNNL: max performance
    elif parsed.sweep_name == 'SHD_SCNN_max_performance':
        print('SHD, SCNN, Hierarchy of kernel sizes')
        config['seed']          = {'values':[0,1,2,3,4]}
        config['use_test_as_valid'] = {'values':[True]}
        config['convolution']   = {'value':True}
        config['hierarchy_conv']= {'value':'kernel'}
        config['conv_dilation'] = {'value':5}
        config['delta_dil']     = {'value':0}
        config['conv_kernel']   = {'value':5}
        config['delta_ker']     = {'value':4}
        config['freq_shift']    = {'value':10}
        config['n_layers']      = {'value':3}
        config['nb_steps']      = {'value':200}
        config['n_hid']         = {'value':128}
        config['dropout_rate']  = {'value':0.4}
        config['l2_lambda']     = {'value':1e-4}
        config['tau_mem']       = {'value':0.02}
        config['n_epochs']      = {'value':100}
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
