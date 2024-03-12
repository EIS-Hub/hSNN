import numpy as np
import argparse

from utils_dataset import get_dataloader
from models import decoder_cum, decoder_sum, decoder_vlast, decoder_vmax
from utils_normalization import BatchNorm, LayerNorm
from utils_initialization import SimArgs, params_initializer
from training import *


if __name__ == '__main__':
    # recover parsed arguments
    parser = argparse.ArgumentParser()
    # network architecture
    parser.add_argument('--n_in', type=int, default=700)
    parser.add_argument('--n_hid', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--normalizer', type=str, default='batch')
    parser.add_argument('--decoder', type=str, default='cum')
    parser.add_argument('--recurrent', type=bool, default=False)
    # time constant
    parser.add_argument('--train_tau', type=bool, default=True)
    parser.add_argument('--hierarchy_tau', type=bool, default=False)
    parser.add_argument('--distrib_tau', type=bool, default=True)
    parser.add_argument('--distrib_tau_sd', type=float, default=0.2)
    parser.add_argument('--tau_mem', type=float, default=0.2)
    parser.add_argument('--delta_tau', type=float, default=0.1)
    # noise
    parser.add_argument('--noise_sd', type=float, default=0.1)
    # regularizers and training args
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--l2_lambda', type=float, default=0)
    parser.add_argument('--freq_lambda', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--verbose', type=bool, default=True)
    parsed = parser.parse_args()
    args = SimArgs(
                 parsed.n_in, parsed.n_hid, parsed.n_layers,
                 parsed.seed, parsed.normalizer, parsed.decoder, 
                 parsed.train_tau, parsed.hierarchy_tau, parsed.distrib_tau,
                 parsed.distrib_tau_sd, parsed.tau_mem, parsed.delta_tau,
                 parsed.noise_sd, parsed.n_epochs, parsed.l2_lambda, 
                 parsed.freq_lambda, parsed.dropout, parsed.recurrent, parsed.verbose
    )

    # Importing the dataset
    train_loader_custom_collate, val_loader_custom_collate, test_loader_custom_collate = get_dataloader( args=args, verbose=True )

    # Adjusting the parameter selection
    # if args.normalizer == 'batch': norm = BatchNorm
    # elif args.normalizer == 'layer': norm = LayerNorm
    # else: norm = None
    # # network architecture
    # if args.recurrent:
    #     layer = rlif_step
    # else: 
    #     layer = lif_step
    # if args.decoder == 'freq':
    #     layer_out = lif_step
    # else: 
    #     layer_out = li_step

    print('\nTraining')
    train_loss, test_acc_shd, val_acc_shd, net_params_trained = train_hsnn(key = jax.random.PRNGKey(args.seed), n_epochs=args.n_epochs, args = args, 
                                                            train_dl = train_loader_custom_collate, test_dl = test_loader_custom_collate, val_dl=val_loader_custom_collate,
                                                            param_initializer=params_initializer, noise_start_step=10, noise_std=0.1, dataset_name='shd', verbose=args.verbose)