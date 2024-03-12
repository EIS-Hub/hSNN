import jax
import time
import numpy as np
import jax.numpy as jnp
from jax.lax import scan
from jax import jit, value_and_grad, vmap
from jax.example_libraries import optimizers

# imports from supporting files
from utils_normalization import LayerNorm, BatchNorm
from models import hsnn, lif_step, rlif_step, li_step, dropout
from models import decoder_cum, decoder_freq, decoder_sum, decoder_vlast, decoder_vmax
from utils_initialization import args


def train_hsnn(key, n_epochs, args, train_dl, test_dl, val_dl, param_initializer, 
                noise_start_step, noise_std, dataset_name, verbose=True):
    
    @jit
    def scan_layer( args_in, input_spikes ):
        args_out_layer, out_spikes_layer = scan( layer, args_in, input_spikes, length=args.nb_steps )
        return args_out_layer, out_spikes_layer
    vscan_layer = vmap( scan_layer, in_axes=(None, 0))

    @jit
    def scan_out_layer( args_in, input_spikes ):
        args_out_layer, out_spikes_layer = scan( layer_out, args_in, input_spikes, length=args.nb_steps )
        return args_out_layer, out_spikes_layer
    vscan_layer_out = vmap( scan_out_layer, in_axes=(None, 0))

    @jit
    def hsnn( args_in, input_spikes ):
        net_params, net_states, key, dropout_rate = args_in
        n_layers = len( net_params )
        # collection of output spikes
        out_spike_net = []
        # Loop over the layers
        for l in range(n_layers):
            if l == 0: layer_input_spike = input_spikes
            else: layer_input_spike = out_spikes_layer
            # making layers' params and states explitic
            # parameters (weights and alpha) and the state of the neurons (spikes, inputs and membrane, ecc..)
            w, alpha = net_params[l]; w_mask, tau, V_mem, out_spikes, v_thr, noise_sd = net_states[l]
            if len(w) == 3: # it means that we'll do normalization
                weight, scale, bias = w
            else: weight = w
            if len(weight) ==2: weight, _ = weight
            # we evolve the state of the neuron according to the LIF formula, Euler approximation
            I_in = jnp.matmul(layer_input_spike, weight)
            # Normalization (if selected)
            if len(w) == 3: # it means that we'll do normalization
                b, t, n = I_in.shape
                I_in = norm( I_in.reshape( b*t, n ), bias = bias, scale = scale )
                I_in = I_in.reshape( b,t,n ) # normalized input current
            # Forward pass of the Layer
            args_in_layer = [net_params[l], net_states[l]]
            if l+1 == n_layers:
                _, out_spikes_layer = vscan_layer_out( args_in_layer, I_in )
            else: 
                _, out_spikes_layer = vscan_layer( args_in_layer, I_in )
                # Dropout
                key, key_dropout = jax.random.split(key, 2)
                out_spikes_layer = dropout( key_dropout, out_spikes_layer, rate=dropout_rate, deterministic=False )
            out_spike_net.append(out_spikes_layer)
        return out_spikes_layer, out_spike_net
    
    decoder_dict = {'cum'  : decoder_cum,
                    'sum'  : decoder_sum,
                    'vmax' : decoder_vmax,
                    'vlast': decoder_vlast,
                    'freq' : decoder_freq
                    }
    decoder = decoder_dict[args.decoder]

    # network architecture
    if args.recurrent:
        layer = rlif_step
    else: 
        layer = lif_step
    if args.decoder == 'freq':
        layer_out = lif_step
    else: 
        layer_out = li_step
    if args.normalizer == 'batch': norm = BatchNorm
    elif args.normalizer == 'layer': norm = LayerNorm
    else: norm = None
    
    key, key_model = jax.random.split(key, 2)

    def loss(key, net_params, net_states, X, Y, epoch, dropout_rate=0.0):
        """ Calculates CE loss after predictions. """

        # we might want to add noise in the forward pass --> memristor-aware-training
        # weight = [net_params[i][0] for i in range( len(net_params) )]
        # weight = cond(
        #     epoch >= noise_start_step, 
        #     lambda weight, key : add_noise(weight, key, noise_std),
        #     lambda weight, key : weight,
        #     weight, key
        # )
        # Forward pass throught the whole model
        args_in = [net_params, net_states, key, dropout_rate]
        output_layer, out_spike_net = hsnn( args_in, X )
        Yhat = decoder( output_layer )
        # Yhat = jax.nn.softmax( net_states_hist[-1][3][:,-1] )
        # compute the loss and correct examples
        num_correct = jnp.sum(jnp.equal(jnp.argmax(Yhat, 1), jnp.argmax(Y, 1)))
        # cross entropy loss
        loss_ce = -jnp.mean( jnp.sum(Y * jnp.log(Yhat+1e-12), axis=-1) )
        # L2 norm
        loss_l2 = optimizers.l2_norm( [net_params[l][0] for l in range(len(net_params))] ) * args.l2_lambda
        # firing rate loss
        avg_spikes_neuron = jnp.mean( jnp.stack( [ jnp.mean( jnp.sum( out_spike_net[l], axis=1 ), axis=(0,-1) ) for l in range( len(net_params)-1 )] ) )
        loss_fr = args.freq_lambda * (args.target_fr - avg_spikes_neuron)**2
        # Total loss
        loss_total = loss_ce + loss_l2 + loss_fr
        loss_values = [num_correct, loss_ce]
        return loss_total, loss_values
 
    @jit
    def update(key, epoch, net_states, X, Y, opt_state, dropout_rate=0.):
        train_params = get_params(opt_state)
        # forward pass with gradients
        value, grads = value_and_grad(loss, has_aux=True, argnums=(1))(key, train_params, net_states, X, Y, epoch, dropout_rate=dropout_rate)
        # possibly disable gradients on alpha and gradient clip
        # for g in range( len( grads )-1 ):
        #     if len(grads[g][0]) == 1:
        #         grads[g][0] = jnp.clip(grads[g][0], -args.grad_clip, args.grad_clip) # weight
        #     if len(grads[g][0]) == 2:
        #         for j in range( len(grads[g][0]) ):
        #             grads[g][0][j] = jnp.clip(grads[g][0][j], -args.grad_clip, args.grad_clip) # weight and recurrent
        #     grads[g][1] = jnp.clip(grads[g][1], -args.grad_clip, args.grad_clip) # alpha
        # grads[-1][0] = jnp.clip(grads[-1][0], -args.grad_clip, args.grad_clip) # weight
        # grads[-1][1] = jnp.clip(grads[-1][1], -args.grad_clip, args.grad_clip) # alpha
        return grads, opt_state, value

    def one_hot(x, n_class):
        return jnp.array(x[:, None] == jnp.arange(n_class), dtype=jnp.float32)

    def total_correct(net_params, net_states, X, Y):
        args_in = [net_params, net_states, key, 0.]
        output_layer, _ = hsnn( args_in, X )
        Yhat = decoder( output_layer )
        # Yhat = jax.nn.softmax( net_states_hist[-1][3][:,-1] )
        acc = jnp.sum(jnp.equal(jnp.argmax(Yhat, 1), Y))
        return acc

    # LR decay
    if args.lr_decay_every + args.lr_start_decay < n_epochs:
        k = int( (args.n_epochs - args.lr_start_decay) // args.lr_decay_every + 1 )
        lr_decay = np.clip( args.lr_decay, 0, 1 )
        # intervals = [ i*args.lr_decay_every for i in range(int((n_epochs)/args.lr_decay_every)-1) ]
        # lr_values = [args.lr*(lr_decay)**i for i in range(int(n_epochs/args.lr_decay_every))]
        intervals = [args.lr_start_decay] + [ args.lr_start_decay+i*args.lr_decay_every for i in range(1,k) ]
        lr_values = [args.lr]+[ args.lr*(lr_decay)**(i+1) for i in range(k) ]
        pw_lr = optimizers.piecewise_constant(intervals, lr_values)
    else: pw_lr = optimizers.piecewise_constant([n_epochs], [args.lr, args.lr*np.clip( args.lr_decay, 1e-5, 1 )])
    # define the optimizer
    opt_init, opt_update, get_params = optimizers.adam(step_size=pw_lr)
    # opt_init, opt_update, get_params = optimizers.sgd(step_size=pw_lr)
    # initialize the parameters (and states)
    net_params, net_states = param_initializer( key_model, args )
    opt_state = opt_init(net_params)


    # Training loop
    train_loss = []
    train_step = 0
    best_val_acc = 5.0; net_params_best = net_params
    for epoch in range(n_epochs):
        t = time.time()
        acc = 0; count = 0
        for batch_idx, (x, y) in enumerate(train_dl):
            y = one_hot(y, args.n_out)
            key, key_epoch = jax.random.split(key)
            _, net_states = param_initializer(key=key_epoch, args=args)
            grads, opt_state, (L, [tot_correct, _]) = update(key, epoch, net_states, x, y, opt_state, dropout_rate=args.dropout_rate)
            # possibly remove gradient from alpha
            if not args.train_alpha:
                for g in range(len(grads)): grads[g][1] *= 0
            # weight update
            opt_state = opt_update(epoch, grads, opt_state)
            net_params = get_params(opt_state)
            # clip alpha between 0 and 1
            # if args.train_alpha:
            #     # for g in range(len(net_params)): net_params[g][1] = jnp.clip(net_params[g][1], jnp.exp(-1/5), jnp.exp(-1/25))
            #     for g in range(len(net_params)): net_params[g][1] = jnp.clip(net_params[g][1], 0, 1.0-1e-5)
            # append stats
            train_loss.append(L)
            train_step += 1
            acc += tot_correct
            count += x.shape[0]
        
        ### Training logs
        train_acc = 100*acc/count
        elapsed_time = time.time() - t            
        ### Validation
        acc_val = 0; count_val = 0
        for batch_idx, (x, y) in enumerate(val_dl):
            count_val += x.shape[0]
            acc_val += total_correct(net_params, net_states, x, y)
        val_acc = 100*acc_val/count_val
        # save best model is validation accuracy is improved
        if val_acc >= best_val_acc: 
            net_params_best = net_params
            best_val_acc = val_acc

        if verbose: print(f'Epoch: [{epoch+1}/{n_epochs}] - Loss: {L:.5f} - '
              f' Training acc: {train_acc:.2f} - Validation acc: {val_acc:.2f} - t: {elapsed_time:.2f} sec')
        # if epoch % 50 == 0:
        #     # Save training state
        #     trained_params = optimizers.unpack_optimizer_state(opt_state)
        #     checkpoint_path = os.path.join('checkpoints', "checkpoint.pkl")
        #     with open(checkpoint_path, "wb") as file:
        #         pickle.dump(trained_params, file)

    # Testing Loop
    acc = 0; val_acc = 0; count = 0
    if dataset_name in ['shd', 'all']:
        for batch_idx, (x, y) in enumerate(val_dl):
            key, key_epoch = jax.random.split(key)
            _, net_states = param_initializer(key=key_epoch, args=args)
            count += x.shape[0]
            acc += total_correct(net_params_best, net_states, x, y)
        val_acc = 100*acc/count
        print(f'Validation Accuracy: {val_acc:.2f}')

    acc = 0; test_acc = 0; count = 0
    if dataset_name in ['shd', 'all']:
        key, key_epoch = jax.random.split(key)
        _, net_states = param_initializer(key=key_epoch, args=args)
        for batch_idx, (x, y) in enumerate(test_dl):
            count += x.shape[0]
            acc += total_correct(net_params_best, net_states, x, y)
        test_acc = 100*acc/count
        print(f'Test Accuracy: {test_acc:.2f}')

    return train_loss, test_acc, val_acc, net_params_best