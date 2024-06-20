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
    # recover parsed arguments
    parser = argparse.ArgumentParser()
    # network architecture
    parser.add_argument('--n_in', type=int, default=700)
    parser.add_argument('--n_hid', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--normalizer', type=str, default='batch')
    parser.add_argument('--decoder', type=str, default='cum')
    parser.add_argument('--recurrent', type=bool, default=False)
    parser.add_argument('--convolution', type=bool, default=False)
    # time constant
    parser.add_argument('--train_alpha', type=bool, default=False)
    parser.add_argument('--hierarchy_tau', type=str, default=False)
    parser.add_argument('--distrib_tau', type=str, default='uniform')
    parser.add_argument('--distrib_tau_sd', type=float, default=0.2)
    parser.add_argument('--tau_mem', type=float, default=0.1)
    parser.add_argument('--delta_tau', type=float, default=0.075)
    # noise
    parser.add_argument('--noise_sd', type=float, default=0.1)
    # regularizers and training args
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--l2_lambda', type=float, default=0)
    parser.add_argument('--freq_lambda', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--dataset_name', type=str, default='shd')
    parser.add_argument('--save_dir_name', type=str, default=None)
    parsed = parser.parse_args()
    print(parsed)
    args = SimArgs(
                 n_in = parsed.n_in, n_hid = parsed.n_hid, n_layers = parsed.n_layers,
                 seed = parsed.seed, normalizer = parsed.normalizer, decoder = parsed.decoder, 
                 train_tau = parsed.train_alpha, hierarchy_tau = parsed.hierarchy_tau, distrib_tau = parsed.distrib_tau,
                 distrib_tau_sd = parsed.distrib_tau_sd, tau_mem = parsed.tau_mem, delta_tau = parsed.delta_tau,
                 noise_sd = parsed.noise_sd, n_epochs = parsed.n_epochs, l2_lambda = parsed.l2_lambda, 
                 freq_lambda = parsed.freq_lambda, dropout = parsed.dropout, recurrent = parsed.recurrent, 
                 verbose = parsed.verbose, save_dir_name=parsed.save_dir_name, dataset_name=parsed.dataset_name,
                 convolution= parsed.convolution
    )

    if args.dataset_name == 'mts_xor':
        args.n_in           = 40
        args.n_out          = 2
        args.n_hid          = 10
        args.decoder        = 'vmem_time'
        args.time_max       = 1.0 # second
        args.timestep       = args.time_max/args.nb_steps # second
        args.tau_out        = 0.05
        args.distrib_tau_sd = 0.1
        args.batch_size     = 512
        # args.normalizer = False
    elif args.dataset_name in ['shd', 'ssc']:
        if args.convolution == True:
            args.nb_steps           = 200 # 250
            args.freq_shift         = 10 #10
            args.use_test_as_valid  = False # True
            args.hierarchy_conv     = 'kernel'
            args.conv_kernel        = 5
            args.delta_ker          = 3
            args.conv_dilation      = 5
            args.delta_dil          = 2
            args.dropout_rate       = 0.4
            args.l2_lambda          = 1e-4
            args.tau_mem            = 0.02
    else: 
        print('Unknown dataset name. Please select a valid task name')


    print('\nTraining')
    train_loss, test_acc, val_acc, net_params_best = train_hsnn( args = args, wandb_flag=False )