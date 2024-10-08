import jax
import jax.numpy as jnp
from jax.lax import scan
from jax import vmap, jit
from functools import partial

# imports from supporting files 
from utils_initialization import args
from utils_normalization import LayerNorm, BatchNorm

### Noise function
@jax.custom_jvp
def add_noise(w, key, noise_std):
    ''' Adds noise only for inference '''
    noisy_w = jnp.where(w != 0.0,
                        w + jax.random.normal(key, w.shape) * jnp.max(jnp.abs(w)) * noise_std,
                        w)
    return noisy_w

@add_noise.defjvp
def add_noise_jvp(primals, tangents):
    weight, key, noise_std = primals
    x_dot, y_dot, z_dot = tangents
    primal_out = add_noise(weight, key, noise_std)
    tangent_out = x_dot
    return primal_out, tangent_out


### Surrogate Gradient function
@jax.custom_jvp
def spiking_fn(x, thr):
    """ Thresholding function for spiking neurons. """
    return (x > thr).astype(jnp.float32)

if args.surrogate_fn == 'box':
    @spiking_fn.defjvp
    def spiking_jpv(primals, tangents):
        """ Surrogate gradient function for thresholding. """
        x, thr = primals
        x_dot, y_dot = tangents
        primal_out = spiking_fn(x, thr)
        # tangent_out = x_dot / (10 * jnp.absolute(x - thr) + 1)**2
        tangent_out = x_dot * ( jnp.absolute(x-thr)<0.5 ).astype(int)
        return primal_out, tangent_out
else: 
    @spiking_fn.defjvp
    def spiking_jpv(primals, tangents):
        """ Surrogate gradient function for thresholding. """
        x, thr = primals
        x_dot, y_dot = tangents
        primal_out = spiking_fn(x, thr)
        tangent_out = x_dot / (10 * jnp.absolute(x - thr) + 1)**2
        return primal_out, tangent_out

def lif_step( args_in, input_spikes ):
    ''' Forward function for the Leaky-Integrate and Fire neuron layer, adopted here for the hidden layers. '''
    net_params, net_states = args_in
    # state: the parameters (weights) and the state of the neurons (spikes, inputs and membrane, ecc..)
    w, alpha = net_params; w_mask, tau, V_mem, out_spikes, v_thr, noise_sd = net_states
    # clip alpha
    alpha = jnp.clip(alpha, 0+1e-5, 1.0-2e-2)

    # V_mem = (alpha) * (V_mem - out_spikes) + (1-alpha) * input_spikes - out_spikes*v_thr
    # V_mem = (alpha) * (V_mem - out_spikes) + I_in #- out_spikes*v_thr
    V_mem = alpha * V_mem + input_spikes - out_spikes*v_thr
    out_spikes = spiking_fn( V_mem, v_thr )
    
    return [ [w, alpha], [w_mask, tau, V_mem, out_spikes, v_thr, noise_sd] ], out_spikes


# Leaky Integrate and Fire layer, Recurrent
def rlif_step( args_in, input_spikes ):
    ''' Forward function for the Leaky-Integrate and Fire neuron layer, adopted here for the hidden layers. '''
    net_params, net_states = args_in
    # state: the parameters (weights) and the state of the neurons (spikes, inputs and membrane, ecc..)
    w, alpha = net_params; w_mask, tau, V_mem, out_spikes, v_thr, noise_sd = net_states
    # clip alpha
    alpha = jnp.clip(alpha, 0+1e-5, 1.0-2e-2)
    win_mask, wrec_mask = w_mask
    if len(w) == 3: # it means that we'll do normalization
        weight, scale, bias = w
    else: weight = w
    win, wrec = weight
    w_rec_diag_zeros = jnp.ones_like(wrec) - jnp.eye( wrec.shape[0] )

    # we evolve the state of the neuron according to the LIF formula, Euler approximation
    I_rec = jnp.matmul(out_spikes, wrec*wrec_mask*w_rec_diag_zeros)
    # V_mem = (alpha) * (V_mem) + (1-alpha) * I_in - out_spikes*v_thr
    V_mem = alpha * V_mem + input_spikes + I_rec - out_spikes*v_thr
    # V_mem = alpha * (V_mem - out_spikes) + (1-alpha) * ( I_in_norm )
    # V_mem = alpha * (V_mem - out_spikes) + (1) * ( I_in_norm )
    out_spikes = spiking_fn( V_mem, v_thr )
    
    return [ [w, alpha], [w_mask, tau, V_mem, out_spikes, v_thr, noise_sd] ], out_spikes

# Leaky Integrator (output layer)
def li_step(args_in, input_spikes):
    ''' Forward function for the Leaky-Integrator neuron layer, adopted here for the output layers. '''
    net_params, net_states = args_in
    # state: the parameters (weights) and the state of the neurons (inputs and membrane)
    w, alpha = net_params; w_mask, tau, V_mem, out_spikes, v_thr, noise_sd = net_states

    # we evolve the state of the neuron according to the LI formula, Euler approximation
    V_mem = (alpha) * (V_mem) + (1-alpha) * input_spikes
    # V_mem = (alpha) * (V_mem) + input_spikes
    
    return [ [w, alpha], [w_mask, tau, V_mem, out_spikes, v_thr, noise_sd] ], V_mem

# 1D Causal Convolution (Delays)
def Conv1D_causal( x, weight, dilation=5 ):
    '''1D Convolution using jax.lax.conv_general_dilated
    from https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html
    x: the input data              [Batch, Length (height), Channels]
    weight: the convolution filter [Kernel size (height), Input channels, Output channels]
        (note: both inputs should be in jnp.float32 type)
    As this is a convolution Causal (only the past influences the future) the input is padded in 
    the left part (beginning) of the input sequence by the same size of the kernel.
    '''
    # turn inputs to the right data-type
    x = x.astype( jnp.float32 )
    # padding for making Conv1D causal
    pad_size = ( (weight.shape[0]-1) * dilation ) +1
    x = jnp.pad( x, pad_width=((0,0),(pad_size-1,0),(0,0)), constant_values=0. )
    # dimention number --> ordering the data for the convolution
    dims=("NHC", "HIO", "NHC")
    rhs_dilation = [dilation]
    return jax.lax.conv_general_dilated( lhs=x, rhs=weight, window_strides=[1], 
                                         padding='valid', dimension_numbers=dims,
                                         rhs_dilation=rhs_dilation )


### old models
# Leaky Integrate and Fire layer
def lif_forward(net_params, net_states, input_spikes, norm=LayerNorm):
    ''' Forward function for the Leaky-Integrate and Fire neuron layer, adopted here for the hidden layers. '''

    # state: the parameters (weights) and the state of the neurons (spikes, inputs and membrane, ecc..)
    w, alpha = net_params; w_mask, tau, V_mem, out_spikes, v_thr, noise_sd = net_states
    if len(w) == 3: # it means that we'll do normalization
        weight, scale, bias = w
    else: weight = w

    # we evolve the state of the neuron according to the LIF formula, Euler approximation
    I_in = jnp.matmul(input_spikes, weight*w_mask)
    # normalize inputs
    if len(w) == 3: I_in_norm = norm( I_in, bias = bias, scale = scale )
    else: I_in_norm = I_in #norm( I_in, bias = jnp.zeros( (I_in.shape[-1]) ), scale = jnp.ones( (I_in.shape[-1]) ) )
    # V_mem = (alpha) * (V_mem - out_spikes) + (1-alpha) * I_in #- out_spikes*v_thr
    # V_mem = (alpha) * (V_mem - out_spikes) + I_in_norm #- out_spikes*v_thr
    V_mem = alpha * V_mem + I_in_norm - out_spikes*v_thr

    out_spikes = spiking_fn( V_mem, v_thr )
    return [w, alpha], [w_mask, tau, V_mem, out_spikes, v_thr, noise_sd]

# Leaky Integrate and Fire layer, Recurrent
def lif_recurrent(net_params, net_states, input_spikes, norm=LayerNorm):
    ''' Forward function for the Leaky-Integrate and Fire neuron layer, adopted here for the hidden layers. '''

    # state: the parameters (weights) and the state of the neurons (spikes, inputs and membrane, ecc..)
    w, alpha = net_params; w_mask, tau, V_mem, out_spikes, v_thr, noise_sd = net_states
    win_mask, wrec_mask = w_mask
    if len(w) == 3: # it means that we'll do normalization
        weight, scale, bias = w
    else: weight = w
    win, wrec = weight
    w_rec_diag_zeros = jnp.ones_like(wrec) - jnp.eye( wrec.shape[0] )


    # we evolve the state of the neuron according to the LIF formula, Euler approximation
    I_in = jnp.matmul(input_spikes, win*win_mask)
    I_rec = jnp.matmul(out_spikes, wrec*wrec_mask*w_rec_diag_zeros)
    # Normalization
    if len(w) == 3: I_in_norm = norm( I_in+I_rec, bias = bias, scale = scale )
    else: I_in_norm = I_in + I_rec #norm( I_in, bias = jnp.zeros( (I_in.shape[-1]) ), scale = jnp.ones( (I_in.shape[-1]) ) )
    # V_mem = (alpha) * (V_mem) + (1-alpha) * I_in - out_spikes*v_thr
    V_mem = alpha * V_mem + I_in_norm - out_spikes*v_thr
    # V_mem = alpha * (V_mem - out_spikes) + (1-alpha) * ( I_in_norm )
    # V_mem = alpha * (V_mem - out_spikes) + (1) * ( I_in_norm )
    out_spikes = spiking_fn( V_mem, v_thr )
    
    return [w, alpha], [w_mask, tau, V_mem, out_spikes, v_thr, noise_sd]

# Leaky Integrator (output layer)
def li_output(net_params, net_states, input_spikes, norm=LayerNorm):
    ''' Forward function for the Leaky-Integrator neuron layer, adopted here for the output layers. '''

    # state: the parameters (weights) and the state of the neurons (inputs and membrane)
    w, alpha = net_params; w_mask, tau, V_mem, out_spikes, v_thr, noise_sd = net_states
    if len(w) == 3:
        # it means that we'll do normalization
        weight, scale, bias = w
    else: weight = w

    # we evolve the state of the neuron according to the LI formula, Euler approximation
    I_in = jnp.matmul(input_spikes, weight*w_mask)
    if len(w) == 3: I_in_norm = norm( I_in, bias = bias, scale = scale )
    else: I_in_norm = I_in #norm( I_in, bias = jnp.zeros( (I_in.shape[-1]) ), scale = jnp.ones( (I_in.shape[-1]) ) )
    V_mem = (alpha) * (V_mem) + (1-alpha) * I_in_norm
    # V_mem = (alpha) * (V_mem) + I_in_norm
    out_spikes = out_spikes + jax.nn.softmax( V_mem, axis=-1 )
    
    return [w, alpha], [w_mask, tau, V_mem, out_spikes, v_thr, noise_sd]

# custom dropout function
def dropout( rng, inputs, rate:float = 0.1, deterministic = True, broadcast_dims = () ):
    # if (rate == 0.) or deterministic:
    #   return inputs
    keep_prob = 1. - rate
    broadcast_shape = list(inputs.shape)
    for dim in broadcast_dims:
      broadcast_shape[dim] = 1
    mask = jax.random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
    mask = jnp.broadcast_to(mask, inputs.shape)
    return jax.lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))

# the SNN network
def hsnn_step( args_in, input_spikes):
    '''The Hierarchical time-constant SNN (hSNN). Made of n_layers layers.'''
    net_params, net_states, key, dropout_rate = args_in
    n_layers = len(net_params)
    for l in range(0, n_layers-1):
        # Hidden layer takes inputs from their previous layer
        if l == 0: input_layer = input_spikes  # First layer takes inputs from "input spikes"
        else: input_layer = net_states[l-1][3] # net_params[l-1][3] : output spikes from previous layer
        net_params[l], net_states[l] = lif_forward( net_params[l], net_states[l], input_layer )
        # dropout
        key, key_dropout = jax.random.split(key, 2)
        net_states[l][3] = dropout( key_dropout, net_states[l][3], rate=dropout_rate, deterministic=False )
    # Output layer is a leaky integrator (LI)
    net_params[-1], net_states[-1] = li_output( net_params[-1], net_states[-1], net_states[-2][3] )

    return [net_params, net_states, key, dropout_rate], net_states # net_params[-1][4] : output leaky membrane voltage

# the SNN network
def hrsnn_step( args_in, input_spikes):
    '''The Hierarchical time-constant SNN (hSNN). Made of n_layers layers.'''
    net_params, net_states, key, dropout_rate = args_in
    n_layers = len(net_params)
    for l in range(0, n_layers-1):
        # Hidden layer takes inputs from their previous layer
        if l == 0: input_layer = input_spikes  # First layer takes inputs from "input spikes"
        else: input_layer = net_states[l-1][3] # net_params[l-1][3] : output spikes from previous layer
        net_params[l], net_states[l] = lif_recurrent( net_params[l], net_states[l], input_layer ) 
        # dropout
        key, key_dropout = jax.random.split(key, 2)
        net_states[l][3] = dropout( key_dropout, net_states[l][3], rate=dropout_rate, deterministic=False )
    # Output layer is a leaky integrator (LI)
    net_params[-1], net_states[-1] = li_output( net_params[-1], net_states[-1], net_states[-2][3] )

    return [net_params, net_states, key, dropout_rate], net_states # net_params[-1][4] : output leaky membrane voltage

def decoder_sum( out_v_mem ):
    ''' Decodes the output as the sum of the membrane voltage over time '''
    # out_v_mem dims: [batch, time_steps, out_dim]
    return jax.nn.softmax( jnp.mean( out_v_mem, axis=1 ), axis=-1 )

def decoder_cum( out_v_mem ):
    ''' Decodes the output as the sum of the "softmaxed" membrane voltage over time '''
    # out_v_mem dims: [batch, time_steps, out_dim]
    return jax.nn.softmax( jnp.sum( jax.nn.softmax( out_v_mem, axis=-1 ), axis=1), axis=-1 )
    # return jnp.sum( jax.nn.softmax( out_v_mem, axis=-1 ), axis=1)

def decoder_vmax( out_v_mem ):
    ''' Decodes the output as the maximum of the membrane voltage over time '''
    # out_v_mem dims: [batch, time_steps, out_dim]
    return jax.nn.softmax( jnp.max( out_v_mem, axis=1 ), axis=-1 )

def decoder_vlast( out_v_mem ):
    ''' Decodes the output as the last value of the membrane voltage over time '''
    # out_v_mem dims: [batch, time_steps, out_dim]
    return jax.nn.softmax( out_v_mem[:,-1], axis=-1 )

def decoder_freq( out_v_mem ):
    ''' Decodes the output as the Frequency of the output neurons over time '''
    # out_v_mem dims: [batch, time_steps, out_dim]
    return jax.nn.softmax( jnp.sum( out_v_mem, axis=1 ), axis=-1 )

def decoder_vmem_time( out_v_mem ):
    ''' Decodes the output as the membrane voltage over time (MTS-XOR) '''
    # out_v_mem dims: [batch, time_steps, out_dim]
    return jax.nn.softmax( out_v_mem, axis=-1 )