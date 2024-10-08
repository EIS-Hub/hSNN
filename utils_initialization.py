import jax
import numpy as np
import jax.numpy as jnp

class SimArgs:    
    def __init__(self, n_in = 700, n_hid = 128, n_layers = 3, 
                 seed=0, normalizer='batch', decoder='cum', 
                 train_tau=False, hierarchy_tau=False, distrib_tau='uniform',
                 distrib_tau_sd=0.2, tau_mem=0.1, delta_tau=0.075,
                 noise_sd=0.1, n_epochs=5, l2_lambda=0,
                 freq_lambda=0, dropout=0.1, recurrent=False, convolution=False,
                 verbose=True, save_dir_name=None, dataset_name = 'shd'):
        # architecture size
        self.dataset_name   = dataset_name
        self.n_in           = n_in # input channels
        self.n_out          = 20 if self.dataset_name =='shd' else 35  # output channels
        self.n_layers       = n_layers # number of layers
        self.n_hid          = n_hid # number of hidden neurons per layer
        # weight
        self.w_scale        = [1/np.sqrt(  float(self.n_in)  )] + [1/np.sqrt( float(self.n_hid) )]*self.n_layers #scaling factors for weight initialization
        self.noise_sd       = 0 # noise to apply to weight during training (not supported for now)
        # input data
        self.nb_rep         = 1
        self.nb_steps       = 100 # number of time steps of the input
        self.time_max       = 1.4 # second
        self.timestep       = self.time_max/self.nb_steps # second
        self.noise_rate     = 0.01 # noise rate, related to a poisson process in the MTS-XOR task
        # input data agumentation
        self.pert_proba     = None # data augmentation
        self.truncation     = False # to use only 150 of 280 timesteps
        self.freq_shift     = 0
        # neuron model
        self.tau_mem        = tau_mem # [second], membrane voltage time constant
        self.tau_out        = 0.2 # [second], membrane voltage time constant (output neurons)
        self.delta_tau      = delta_tau # [second], different of tau between input and output layers
        self.v_rest         = 0 # resting membrane voltage
        self.v_thr          = 1 # threshold voltage
        self.v_reset        = 0 # reset voltage
        self.surrogate_fn   = 'box' # type of surrogate gradient
        # network
        self.decoder        = decoder # output decoding strategy
        self.recurrent      = recurrent # enables recurrent connections
        self.convolution    = convolution
        self.conv_kernel    = 5 #[5,5,5,8]  #[5]*self.n_layers
        self.conv_dilation  = 5 #[4]*self.n_layers
        self.hierarchy_conv = False
        self.delta_ker      = 0
        self.delta_dil      = 0
        self.distrib_tau    = distrib_tau # enables individual time constant per neuron
        self.distrib_tau_sd = distrib_tau_sd # standard dev of the time constant distribution
        self.hierarchy_tau  = hierarchy_tau # enables hierarchy of time constants
        self.tanh_coef      = 0.5 # sets the steepness of the tanh function through the hidden layers
        self.tanh_center    = 0.5 # sets the "zero" (reference) for the tanh function scaling the time constant
        self.train_alpha    = train_tau # enables training the time constant
        self.normalizer     = normalizer # selects the normalization layer
        self.norm_bias_init = 0.0 
        # training
        self.lr             = 0.01 # 0.01 # learning rate
        self.n_epochs       = n_epochs # number of epochs
        self.grad_clip      = 1000 # gradient clipping
        self.batch_size     = 256 # batch size
        self.seed           = seed # seed, for reproducibility
        self.lr_config      = 2 
        self.lr_decay       = 0.5 #0.75 # learning rate decay
        self.lr_decay_every = 5 #10
        self.lr_start_decay = 25
        self.l2_lambda      = l2_lambda
        self.l2_alpha_sd    = 1e-2 #0 # penalization on the standard dev of the time constants
        self.freq_lambda    = freq_lambda
        self.target_fr      = 12.
        self.dropout_rate   = dropout
        self.verbose        = verbose
        self.use_test_as_valid = False # be very careful. This flag selects the training set as the validation set. 
                                       # It's a really bad practise, but it has unfortunately become standard in SHD...
                                       # see Bittar 2022, or Masquelier 2022...
        # saving options
        self.save_dir_name  = save_dir_name
        self.wandb          = True
        self.experiment_name= 'hssn_test'
args = SimArgs()

# function that initializes the hyperparameters of the network
def params_initializer( key, args ):
    """ Initialize parameters. """
    key_hid = jax.random.split(key, args.n_layers); key=key_hid[0]; key_hid=key_hid[1:]

    # initialize the time constants
    if args.hierarchy_tau == 'tanh':
        tanh = lambda x, a: (np.exp(2*x/a)-1)/(np.exp(2*x/a)+1)
        rescale = lambda x: ( x - np.min(x) ) / (np.max( np.abs( x - np.min(x) ) )) - 0.5
        hierarchy = tanh( (np.linspace(0,1,args.n_layers-1)-args.tanh_center), args.tanh_coef )
        scaled_hierarchy = rescale( hierarchy ) * args.delta_tau
        tau_layer_list = args.tau_mem + scaled_hierarchy
    elif args.hierarchy_tau == 'linear':
        tau_start = np.clip(args.tau_mem - args.delta_tau * 0.5, 0, None) # [second] input time constant
        tau_end   = np.clip(args.tau_mem + args.delta_tau * 0.5, 0, None) # [second] output time constant
        tau_layer_list = jnp.linspace( tau_start, tau_end, args.n_layers-1 )
    else: 
        tau_layer_list = jnp.ones( args.n_layers-1 )*args.tau_mem
    
    # Hierarchy on the convolution
    if args.hierarchy_conv == 'dilation':
        # dilations
        # args.delta_dil = args.delta_dil
        # kernels
        args.conv_kernels = np.ones( args.n_layers ).astype(int)*args.conv_kernel
    elif args.hierarchy_conv == 'kernel': 
        # kernels
        ker_start = np.clip( int( args.conv_kernel - args.delta_ker * 0.5 ), 1, None) # initial layer dilation
        ker_end = np.clip( int( args.conv_kernel + args.delta_ker * 0.5 ), 1, None) # final layer dilation
        ker_layer_list = np.linspace( ker_start, ker_end, args.n_layers-1 ).astype(int)
        ker_layer_list = np.pad( ker_layer_list, (0, 1), constant_values=args.conv_kernel )
        args.conv_kernels = ker_layer_list
        # dilations
    elif args.hierarchy_conv == 'both':
        # kernels
        ker_start = np.clip( int( args.conv_kernel - args.delta_ker * 0.5 ), 1, None) # initial layer dilation
        ker_end = np.clip( int( args.conv_kernel + args.delta_ker * 0.5 ), 1, None) # final layer dilation
        ker_layer_list = np.linspace( ker_start, ker_end, args.n_layers-1 ).astype(int)
        ker_layer_list = np.pad( ker_layer_list, (0, 1), constant_values=args.conv_kernel )
        args.conv_kernels = ker_layer_list
        # dilations
        # args.delta_dil = args.delta_dil
    else:
        args.conv_kernels = np.ones( args.n_layers ).astype(int)*args.conv_kernel

    # Initializing the weights, weight masks and time constant (alpha factors)
    w_scale = reshape_weight_scale_factor(args.w_scale, args.n_layers, args.recurrent)
    net_params, net_states = [], []
    for l in range(args.n_layers):

        if l == args.n_layers-1:
            n_pre = args.n_hid; n_post = args.n_out
            # same time-constant for output neurons
            tau_l = args.tau_out #args.tau_start + (l/args.n_layers)*( args.tau_end-args.tau_start )
            alpha_l = jnp.exp(-args.timestep/tau_l)

        else:
            if l == 0: n_pre = args.n_in; n_post = args.n_hid
            else: n_pre = args.n_hid; n_post = args.n_hid

            # partition of the time constants in the different layers
            # if args.hierarchy_tau: tau_layer = args.tau_start + (l/ (args.n_layers-1))*( args.tau_end-args.tau_start )
            # else: tau_layer = args.tau_mem
            tau_layer = tau_layer_list[l]
            if args.distrib_tau == 'uniform':
                tau_l = jax.random.uniform(key_hid[l], [args.n_hid], 
                                           minval=tau_layer*(1-args.distrib_tau_sd), 
                                           maxval=tau_layer*(1+args.distrib_tau_sd)  )
                tau_l = jnp.clip( tau_l, 1e-10, 10 ) # clipping to avoid too short/long time constant
            elif args.distrib_tau == 'normal':
                tau_l = jax.random.normal(key_hid[l], [args.n_hid]) * args.distrib_tau_sd * tau_layer + tau_layer
                tau_l = jnp.clip( tau_l, 1e-10, 10 ) # clipping to avoid too short/long time constant
            else:
                tau_l = tau_layer
            alpha_l = jnp.exp(-args.timestep/tau_l)
            alpha_l = jnp.clip( alpha_l, 0.1, 0.99 )

        # initializing the hidden weights with a normal distribution
        if not args.recurrent: w_scale_ff = w_scale[l]
        else: w_scale_ff = w_scale[l][0]
        if args.convolution and l!=(args.n_layers-1):
                weight_l = jax.random.uniform(key_hid[l], [args.conv_kernels[l], n_pre, n_post], 
                                              minval=-w_scale_ff/(0.5*np.sqrt(args.conv_kernels[l])), 
                                              maxval=w_scale_ff/(0.5*np.sqrt(args.conv_kernels[l])))
        else:
            weight_l = jax.random.uniform(key_hid[l], [n_pre, n_post], minval=-w_scale_ff, maxval=w_scale_ff)
        # weight_l = jax.random.normal(key_hid[l], [n_pre, n_post]) * w_scale_ff
        weight_mask_l = 1 # jax.random.uniform(key_hid[l], [n_pre, n_post]) < (1/args.n_layers)
        if args.recurrent and l!=args.n_layers-1 : 
            weight_rec_l = jax.random.uniform(key_hid[l], [n_post, n_post], minval=-w_scale[l][1], maxval=w_scale[l][1])
            # weight_rec_l = jax.random.normal(key_hid[l], [n_post, n_post]) * w_scale[l][1]
            weight_l = [weight_l, weight_rec_l]
            weight_mask__rec_l = 1 # jax.random.uniform(key_hid[l], [n_post, n_post]) < (1/args.n_layers)
            weight_mask_l = [weight_mask_l, weight_mask__rec_l]
        if args.normalizer in ['batch', 'layer']:
            scale_norm = jnp.ones( (n_post) ) #jax.random.normal(key_hid[l], [n_post]) * (1/args.w_scale)
            bias_norm = jnp.zeros( (n_post) ) + args.norm_bias_init #jax.random.normal(key_hid[l], [n_post]) * (1/args.w_scale)
            weight_l = [weight_l, scale_norm, bias_norm]

        # the initialization of the membrane voltage
        v_mems = jnp.zeros( (n_post) ) #jax.random.uniform(key_hid[l], [n_post], minval=0., maxval=0.25 ) #jnp.zeros( (n_post) )
        if l == args.n_layers -1:
            out_spikes = jnp.zeros( (n_post) ) #jax.random.uniform(key_hid[l], [n_post], minval=0., maxval=0.99 ) #jnp.zeros( (n_post) )
        else: out_spikes = jnp.zeros( (n_post) ) # jax.random.uniform(key_hid[l], [n_post], minval=0., maxval=0. )

        # building the parameters for each layer
        net_params.append( [weight_l, alpha_l] )
        net_states.append( [weight_mask_l, tau_l, v_mems, out_spikes, args.v_thr, args.noise_sd] )

    return net_params, net_states

def reshape_weight_scale_factor( w_scale, n_layers, recurrent = False ):
    w_scale = jnp.array(w_scale)
    if recurrent: 
        w_scale_out = jnp.zeros( (n_layers, 2) )
        if w_scale.shape == (n_layers, 2): w_scale_out = w_scale
        elif w_scale.shape == (n_layers,): w_scale_out = jnp.stack( [w_scale, w_scale] ).reshape(2, n_layers).T
        elif w_scale.shape == (n_layers*2,): w_scale_out = jnp.stack( [w_scale] ).reshape(2, n_layers).T
        elif w_scale.shape == (2,): w_scale_out = jnp.array(list(w_scale)*n_layers).reshape(n_layers, 2)
        elif w_scale.shape == (1,): w_scale_out = jnp.array(list(w_scale)*n_layers*2).reshape(n_layers, 2)
        else: w_scale_out = jnp.array( [ jnp.mean(w_scale) ]*n_layers*2).reshape(n_layers, 2)
    else:
        w_scale_out = jnp.zeros( (n_layers) )
        if w_scale.shape == (n_layers, ): w_scale_out = w_scale
        elif w_scale.shape == (1,): w_scale_out = jnp.array(list(w_scale)*n_layers).reshape(n_layers)
        else: w_scale_out = jnp.array( [ jnp.mean(w_scale) ]*n_layers).reshape(n_layers)
    return w_scale_out