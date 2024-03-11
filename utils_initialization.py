import jax
import jax.numpy as jnp

class SimArgs:
    def __init__(self):
        # archi
        self.n_in = 700
        self.n_out = 20
        self.n_layers = 3
        self.n_hid = 128
        # weight
        self.w_scale = 0.3
        self.pos_w = False # use only positive weights at initizialization
        self.noise_sd = 0 # [0.05, 0.1, 0.15, 0.2]
        # data
        self.nb_rep = 1
        self.nb_steps = 100 #int( np.round( self.time_max/self.timestep, 0 ) )
        self.time_max = 1.4 # second
        self.timestep = self.time_max/self.nb_steps # 0.014 #0.005 # second, 280 timesteps
        self.pert_proba = None
        self.truncation = False # to use only 150 of 280 timesteps 
        # neuron model
        self.tau_start = 4*self.timestep # second
        self.tau_end   = self.time_max/4 # second
        self.tau_mem = 40e-3
        self.tau_out = 1e-3
        self.recurrent = False
        self.distrib_tau = True
        self.distrib_tau_bittar = False
        self.distrib_tau_sd = 0.6
        self.hierarchy_tau = True
        self.train_alpha = True
        self.normalizer = 'layer'
        self.norm_bias_init = 0.2
        self.v_rest = 0 
        self.v_thr = 1
        self.v_reset = 0
        self.surrogate_fn = 'box'
        # training
        self.lr = 0.01
        self.nb_epochs = 5
        self.grad_clip = 1000
        self.batch_size = 128
        self.seed = 42
        self.lr_config = 2 
        self.lr_decay = 0.5
        self.lr_decay_every = 25
        self.l2_lambda = 1e-5
        self.freq_lambda = 1e-7
        self.target_fr = 12.
        self.dropout_rate = 0.1
args = SimArgs()

# function that initializes the hyperparameters of the network
def params_initializer( key, args ):
    """ Initialize parameters. """
    key_hid = jax.random.split(key, args.n_layers); key=key_hid[0]; key_hid=key_hid[1:]

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
            if args.hierarchy_tau: tau_layer = args.tau_start + (l/ (args.n_layers-1))*( args.tau_end-args.tau_start )
            else: tau_layer = args.tau_mem
            if args.distrib_tau:
                # tau_l = jax.random.uniform(key_hid[l], [args.n_hid], minval=np.exp(-1/5), maxval=np.exp(-1/25)  )
                tau_l = jax.random.uniform(key_hid[l], [args.n_hid], minval=tau_layer*(1-args.distrib_tau_sd), maxval=tau_layer*(1+args.distrib_tau_sd)  )
                # tau_l = jax.random.normal(key_hid[l], [args.n_hid]) * args.distrib_tau_sd * tau_layer + tau_layer
                tau_l = jnp.clip( tau_l, 0.005, 10 ) # clipping to avoid too short/long time constant
                if args.distrib_tau_bittar: tau_l = jax.random.uniform(key_hid[l], [args.n_hid], minval=args.timestep*5, maxval=args.timestep*25  )
            else:
                tau_l = tau_layer
            alpha_l = jnp.exp(-args.timestep/tau_l)
            # alpha_l = jax.random.normal( key_hid[l], [args.n_hid] ) * jnp.exp( -args.timestep/tau_layer ) * args.distrib_tau_sd + jnp.exp( -args.timestep/tau_layer )
            alpha_l = jnp.clip( alpha_l, 0.5, 0.99 )

        # initializing the hidden weights with a normal distribution
        if not args.recurrent: w_scale_ff = w_scale[l]
        else: w_scale_ff = w_scale[l][0]
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