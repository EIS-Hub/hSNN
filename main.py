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

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".5" # needed because network is huge
os.environ["CUDA_VISIBLE_DEVICES"]="1"
jax.devices()

if __name__ == '__main__':
    # recover parsed arguments
    parser = argparse.ArgumentParser()
    # network architecture
    parser.add_argument('--n_in', type=int, default=700)
    parser.add_argument('--n_hid', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--normalizer', type=str, default='batch')
    parser.add_argument('--decoder', type=str, default='cum')
    parser.add_argument('--recurrent', type=bool, default=False)
    # time constant
    parser.add_argument('--train_alpha', type=bool, default=False)
    parser.add_argument('--hierarchy_tau', type=bool, default=False)
    parser.add_argument('--distrib_tau', type=str, default='uniform')
    parser.add_argument('--distrib_tau_sd', type=float, default=0.2)
    parser.add_argument('--tau_mem', type=float, default=0.1)
    parser.add_argument('--delta_tau', type=float, default=0.05)
    # noise
    parser.add_argument('--noise_sd', type=float, default=0.1)
    # regularizers and training args
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--l2_lambda', type=float, default=0)
    parser.add_argument('--freq_lambda', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--save_dir_name', type=str, default=None)
    parsed = parser.parse_args()
    print(parsed)
    args = SimArgs(
                 n_in = parsed.n_in, n_hid = parsed.n_hid, n_layers = parsed.n_layers,
                 seed = parsed.seed, normalizer = parsed.normalizer, decoder = parsed.decoder, 
                 train_tau = parsed.train_alpha, hierarchy_tau = parsed.hierarchy_tau, distrib_tau = parsed.distrib_tau,
                 distrib_tau_sd = parsed.distrib_tau_sd, tau_mem = parsed.tau_mem, delta_tau = parsed.delta_tau,
                 noise_sd = parsed.noise_sd, n_epochs = parsed.n_epochs, l2_lambda = parsed.l2_lambda, 
                 freq_lambda = parsed.freq_lambda, dropout = parsed.dropout, recurrent = parsed.recurrent, verbose = parsed.verbose, save_dir_name=parsed.save_dir_name
    )

    print('\nTraining')
    train_loss, test_acc, val_acc, net_params_best = train_hsnn( args = args, dataset_name='shd')
    