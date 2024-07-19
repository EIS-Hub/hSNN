import matplotlib.pyplot as plt
import numpy as np
import os
import jax
import jax.numpy as jnp
from scipy.stats import lognorm, norm
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import random
import urllib.request
import gzip, shutil
import hashlib
import h5py
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve

from utils_initialization import SimArgs
args = SimArgs()

def get_audio_dataset(cache_dir, cache_subdir, dataset_name):
    # The remote directory with the data files
    base_url = "https://zenkelab.org/datasets"

    # Retrieve MD5 hashes from remote
    response = urllib.request.urlopen(f"{base_url}/md5sums.txt")
    data = response.read()
    lines = data.decode('utf-8').split("\n")
    file_hashes = {line.split()[1]: line.split()[0] for line in lines if len(line.split()) == 2}

    # Download the Spiking Heidelberg Digits (SHD) dataset
    if dataset_name == 'shd':
        files = [ "shd_train.h5.gz", "shd_test.h5.gz"]
    if dataset_name == 'ssc':
        files = [ "ssc_train.h5.gz", "ssc_valid.h5.gz", "ssc_test.h5.gz"]

    for fn in files:
        origin = f"{base_url}/{fn}"
        hdf5_file_path = get_and_gunzip(origin, fn, md5hash=file_hashes[fn], cache_dir=cache_dir, cache_subdir=cache_subdir)
        # print(f"File {fn} decompressed to:")
        print(f"Available at: {hdf5_file_path}")

def get_and_gunzip(origin, filename, md5hash=None, cache_dir=None,
                   cache_subdir=None):
    gz_file_path = get_file(filename, origin, md5_hash=md5hash,
                            cache_dir=cache_dir, cache_subdir=cache_subdir)
    hdf5_file_path = gz_file_path[:-3]
    if not os.path.isfile(hdf5_file_path) or \
            os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
        print(f"Decompressing {gz_file_path}")
        with gzip.open(gz_file_path, 'r') as f_in, \
                open(hdf5_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path

def get_numpy_datasets(subkey_perturbation, pert_proba, n_inp, cache_dir, nb_steps, freq_shift=0, dataset_name='shd', download=False, truncation=False):
    cache_subdir = f"audiospikes"
    if download:
        get_audio_dataset(cache_dir, cache_subdir, dataset_name)

    train_ds = []; test_ds = []
    if dataset_name in ['shd', 'ssc']:
        train_file = h5py.File(os.path.join(cache_dir, cache_subdir,
                                                dataset_name+'_train.h5'
                                                ), 'r')
        test_file  = h5py.File(os.path.join(cache_dir, cache_subdir,
                                                dataset_name+'_test.h5'
                                                ), 'r')
        _train_ds = DatasetNumpy(train_file['spikes'],
                                    train_file['labels'],
                                    name=dataset_name, target_dim=n_inp, nb_rep=1,
                                    nb_steps=nb_steps, pert_proba=pert_proba, 
                                    subkey_perturbation=subkey_perturbation, 
                                    truncation=truncation, freq_shift=freq_shift) 
        _test_ds  = DatasetNumpy(test_file['spikes'],
                                    test_file['labels'],
                                    name=dataset_name, target_dim=n_inp, nb_rep=1, ################################################ nb_rep
                                    nb_steps=nb_steps, truncation=truncation) 
        if dataset_name == 'ssc':
            valid_file = h5py.File(os.path.join(cache_dir, cache_subdir,
                                                dataset_name+'_valid.h5'
                                                ), 'r')
            _valid_ds = DatasetNumpy(valid_file['spikes'],
                                    valid_file['labels'],
                                    name=dataset_name, target_dim=n_inp, nb_rep=1,
                                    nb_steps=nb_steps, pert_proba=pert_proba, 
                                    subkey_perturbation=subkey_perturbation, 
                                    truncation=truncation) 
            
            train_ds.append( [_train_ds, _valid_ds] )
        else: train_ds.append( _train_ds )
    test_ds.append(_test_ds)
    return train_ds, test_ds

class DatasetNumpy(torch.utils.data.Dataset):
    """
    Numpy based generator
    """
    def __init__(self, spikes, labels, name, target_dim, nb_rep, nb_steps, freq_shift=0, pert_proba=None, subkey_perturbation=None, truncation=False, verbose=False):
        if verbose: print(pert_proba, subkey_perturbation)
        self.nb_steps = nb_steps # number of time steps in the input
        # print(f'nb_step: {self.nb_steps} (DatasetNumpyModified.__init__)')
        self.nb_units = 700   # number of input units (channels)
        self.max_time = 1.4   # maximum recording time of a digit (in s)
        self.spikes = spikes  # recover the 'spikes' dictionary from the h5 file
        self.labels = labels  # recover the 'labels' array from the h5 file
        self.name = name      # name of the dataset or name of speaker
        if freq_shift != 0:
            self.freq_shifting_augment = True
            self.freq_shift = freq_shift
        else:
            self.freq_shifting_augment = False
            self.freq_shift = 0

        # Loading the dataset
        self.firing_times = self.spikes['times']
        self.units_fired  = self.spikes['units']
        self.num_samples = self.firing_times.shape[0]
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)

        # initialize the input (3D) and output (1D) arrays
        self.input  = np.zeros((self.num_samples, self.nb_steps,
                                 self.nb_units), dtype=np.uint8)
        self.output = np.array(self.labels, dtype=np.uint8)

        # Loading the dataloader
        self.load_spikes()

        # Reducing the dimention of the input
        if target_dim != 700:
            self.reduce_inp_dimensions(target_dim=target_dim, axis=2, nb_rep=nb_rep)

        # perform trunctation (not used in this code)
        if truncation and verbose:
            self.input = self.input[:, :150,:]
            print(f'TRUNCATION: ON')
            print(f'nb_step after truncation: {self.input.shape[1]} (DatasetNumpyModified.__init__)')
        elif verbose: 
            print(f'TRUNCATION: OFF')

        # Introduce noise in form of added spikes (not used in this code)
        if pert_proba != None:
            perturbation = jax.random.bernoulli(subkey_perturbation, p=pert_proba, shape=self.input.shape)
            perturbation = jnp.logical_or(self.input, perturbation).astype(jnp.uint8)
            self.input = jnp.concatenate([self.input, perturbation], axis=0, dtype=jnp.uint8)
            self.output = jnp.tile(self.output, reps=2)
            if verbose: print(f'self.input after perturbation: {self.input.shape} (DatasetNumpyModified.__init__)')
        self.num_samples = self.input.shape[0]

    def __len__(self):
        return self.num_samples

    def load_spikes(self):
        """
        For each sample, we create a 2D array of size (nb_steps, nb_units).
        We downsample the firing times and the units fired to the time bins
        :return:
        """
        for idx in range(self.num_samples):
            times = np.digitize(self.firing_times[idx], self.time_bins)
            units = self.units_fired[idx]
            # self.input[idx, times, units] = 1

            x_idx = torch.LongTensor(np.array([times, units])).to('cpu')
            x_val = torch.FloatTensor(np.ones(len(times))).to('cpu') # torch.sparse_coo_tensor(indices, values, shape, dtype=, device=)
            x_size = torch.Size([self.nb_steps, self.nb_units])

            x = torch.sparse_coo_tensor(x_idx, x_val, x_size).to('cpu')
            # y = self.labels[idx]

            self.input[idx] = x.to_dense()
            # return x.to_dense(), y


    def reduce_inp_dimensions(self, target_dim, axis, nb_rep):
        '''We take the original dataset featuring 700 inputs and we rescale it to "target_dim" < 700'''
        sample_ind = int(np.ceil(self.nb_units / target_dim))
        assert nb_rep <= sample_ind, f'The maximum factor of data augmentation is {sample_ind}, you provided {nb_rep}'
        index = [np.arange(i, 700, sample_ind) for i in range(sample_ind)]
        reshaped = [np.take(self.input, index[i], axis)
                    for i in range(nb_rep)] # this samples the data a
        reshaped = [np.pad(reshaped[i],
                            [(0, 0), (0, 0),
                             (0, int(target_dim-reshaped[i].shape[2]))],
                            mode='constant')
                    for i in range(nb_rep)]
        reshaped = np.concatenate(reshaped, axis=0)

        self.input = reshaped
        self.output = np.tile(self.output, nb_rep)
        self.num_samples = reshaped.shape[0]

    def __getitem__(self, idx):
        inputs, outputs = self.__data_generation(idx)
        return inputs, outputs

    def __data_generation(self, idx):
        if self.name == 'shd':
            output = self.output[idx]
        if self.name == 'ssc':
            output = self.output[idx]
        return self.input[idx], output

def frequency_shift( input, freq_shift=20, p_top=0.5 ):
    '''We take the original dataset (700) inputs and pad either the top or bottom with zeros, then cropping the exceeding frequency channels
    - frequency_shift : number of channels to shift up or down
    - p_top           : probability to add the frequency shit from the top (= 1 - p_bot) 
    '''
    in_size = input.shape[0]
    input_padded = np.copy(input)
    freq_shift = np.clip( int(np.random.normal()*3 + freq_shift), 0, None )
    up_idx = np.random.uniform( size=in_size ) >= p_top
    input_padded[up_idx] = np.pad( input[up_idx], ((0,0),(0,0),(freq_shift,0)) )[:,:,:input.shape[2]]
    input_padded[~up_idx] = np.pad( input[~up_idx], ((0,0),(0,0),(0,freq_shift)) )[:,:,freq_shift:]
    return input_padded

def get_file(fname,
             origin,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.data-cache')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.data-cache')
    datadir = os.path.join(datadir_base, cache_subdir)

    # Create directories if they don't exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(datadir, exist_ok=True)

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
    # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the ' + hash_algorithm +
                      ' file hash does not match the original value of ' + file_hash +
                      ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)

    return fpath

def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    if (algorithm == 'sha256') or \
            (algorithm == 'auto' and len(file_hash) == 64):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()

def custom_collate_fn(batch): 
  transposed_data = list(zip(*batch))

  labels = np.array(transposed_data[1])
  spikes = np.array(transposed_data[0])

  return spikes, labels

# Multi-Time-Scale XOR dataset (MTS_XOR)
class DatasetNumpy_MTS_XOR(torch.utils.data.Dataset):
    """
    Creating a Numpy-based Dataset for the MTS-XOR task.
    Details on the task here: https://www.nature.com/articles/s41467-023-44614-z
    """
    def __init__(self, batch_size=256, n_epochs=150, nb_steps=100, 
                 channel=2, channel_rate=[0.2, 0.6], 
                 noise_rate=0.01, channel_size=20,
                 coding_time=10, remain_time=5, 
                 start_time=10, verbose=False):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.nb_steps = nb_steps  # number of time steps in the input
        self.channel = channel
        self.channel_rate = channel_rate
        self.noise_rate = noise_rate
        self.channel_size = channel_size
        self.coding_time = coding_time
        self.remain_time = remain_time
        self.start_time = start_time

        #init xor label
        self.label = torch.zeros(len(channel_rate),len(channel_rate))
        self.label[1][0] = 1
        self.label[0][1] = 1

        # initialize the input (3D) and output (2D) arrays
        x_train, y_train = [], []
        for e in range(self.n_epochs):
            x, y = self.get_batch()
            x_train.append( x ); y_train.append(y)
        self.input = np.vstack(x_train).astype(np.int8) 
        self.output = np.vstack(y_train).astype(np.int8)
        self.num_samples = self.input.shape[0]

    def __len__(self):
        return self.num_samples

    def get_batch(self):
        """Generate the mutlitimescale spiking xor problem batch"""
        # Build the first sequence
        values = torch.rand(self.batch_size,self.nb_steps,self.channel_size*2,requires_grad=False) <= self.noise_rate
        targets = torch.zeros(self.nb_steps,self.batch_size,requires_grad=False).int()
        #build the signal 1
        init_pattern = torch.randint(len(self.channel_rate),size=(self.batch_size,))
        #generate spikes
        prob_matrix = torch.ones(self.start_time,self.channel_size,self.batch_size)*torch.tensor(self.channel_rate)[init_pattern]
        add_patterns = torch.bernoulli(prob_matrix).permute(2,0,1).bool()
        values[:,:self.start_time,:self.channel_size] = values[:,:self.start_time,:self.channel_size] | add_patterns
        
        #build the signal 2
        for i in range((self.nb_steps-self.start_time) //(self.coding_time+self.remain_time)):
            pattern = torch.randint(len(self.channel_rate),size=(self.batch_size,))
            label_t = self.label[init_pattern,pattern].int()
            #generate spikes
            prob = torch.tensor(self.channel_rate)[pattern]
            prob_matrix = torch.ones(self.coding_time,self.channel_size,self.batch_size)*prob
            add_patterns = torch.bernoulli(prob_matrix).permute(2,0,1).bool()

            values[:,self.start_time+i*(self.coding_time+self.remain_time)+self.remain_time:self.start_time+(i+1)*(self.coding_time+self.remain_time),self.channel_size:] = values[:,self.start_time+i*(self.coding_time+self.remain_time)+self.remain_time:self.start_time+(i+1)*(self.coding_time+self.remain_time),self.channel_size:] | add_patterns
            targets[self.start_time+i*(self.coding_time+self.remain_time):self.start_time+(i+1)*(self.coding_time+self.remain_time)] = label_t
        return values.detach().numpy(), targets.transpose(0,1).contiguous().detach().numpy()

    def __getitem__(self, idx):
        inputs, outputs = self.__data_generation(idx)
        return inputs, outputs

    def __data_generation(self, idx):
        return self.input[idx], self.output[idx]
    
def current2firing_time(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    """ Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.

    Args:
    x -- The "current" values

    Keyword args:
    tau -- The membrane time constant of the LIF neuron to be charged
    thr -- The firing threshold value 
    tmax -- The maximum time returned 
    epsilon -- A generic (small) epsilon > 0

    Returns:
    Time to first spike for each "current" x

    --> used for MNIST
    """
    idx = x<thr
    x = np.clip(x,thr+epsilon,1e9)
    T = tau*np.log(x/(x-thr))
    T[idx] = tmax
    return T

def expand_time_dim( data, nb_steps=100 ):
    ''' This function implements the latency coding encoding of a static (gray-scale image-like) dataset
            -data: the input data, in [batch, flattened_image_size]
            -nb_steps: the time dimension size to add
        The gray-scale entries are first discretized and then then corresponding temporal entry is set to "1"
        All other temporal entries are set to 0.
    '''
    times = current2firing_time( data )
    data_time = torch.zeros( (data.shape[0], nb_steps, data.shape[-1]), dtype=torch.uint8 )
    bs, ids = torch.where( times.type(torch.int8) != 1 )
    data_time[bs, torch.clip( times.type(torch.int)[bs, ids], 0, nb_steps-1).type(torch.int), ids] = 1
    return  data_time

def get_dataloader( args, 
                    cache_dir='/Users/filippomoro/Desktop/KINGSTONE/Datasets/SHD',
                    # cache_dir='/home/ttorchet/data',
                    download=False,
                    verbose=False):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # selecting KeyWord Spotting tasks
    if args.dataset_name in ['shd', 'ssc']:
        # cache_dir = '/Users/filippomoro/Desktop/KINGSTONE/Datasets/SHD' # take data from tristan, to avoid copies #os.getcwd()
        if os.getcwd() == '/home/filippo/hsnn':
            cache_dir = '/home/filippo/data'
        elif os.getcwd() == '/Users/filippomoro/Documents/hsnn':
            cache_dir = '/Users/filippomoro/Desktop/KINGSTONE/Datasets/SHD'
        else: cache_dir = cache_dir
        key = jax.random.PRNGKey(args.seed)
        key, subkey_perturbation = jax.random.split(key)
        train_ds, test_ds = get_numpy_datasets(subkey_perturbation, args.pert_proba, args.n_in, 
                                            cache_dir=cache_dir, download=download,
                                            dataset_name=args.dataset_name, freq_shift=args.freq_shift,
                                            nb_steps=args.nb_steps, truncation=args.truncation)

        # Validation set splitting (if not explicit in dataset)
        train_ds = train_ds[0]
        test_ds = test_ds[0]
        if args.dataset_name == 'ssc':
            train_ds_split, val_ds_split = train_ds
        else:
            train_size = int(0.8 * len(train_ds))
            val_size   = len(train_ds) - train_size
            train_ds_split, val_ds_split = random_split(train_ds, [train_size, val_size])
        if verbose:
            print(f'Train DL size: {len(train_ds_split)}, Validation DL size: {len(val_ds_split)}, Test DL size: {len(test_ds)}')

        train_loader_custom_collate = DataLoader(train_ds_split, args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader_custom_collate   = DataLoader(val_ds_split,   args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        if args.dataset_name == 'ssc':
            test_loader_custom_collate  = DataLoader(test_ds,        args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        else: test_loader_custom_collate  = DataLoader(test_ds,        args.batch_size, shuffle=None, collate_fn=custom_collate_fn)
        return train_loader_custom_collate, val_loader_custom_collate, test_loader_custom_collate
    
    # selecting the MTS-XOR task
    elif args.dataset_name == 'mts_xor':
        # get the Dataset
        train_ds_split = DatasetNumpy_MTS_XOR( n_epochs=20, batch_size=args.batch_size, nb_steps=args.nb_steps, channel_size=int(args.n_in/2), noise_rate=args.noise_rate )
        val_ds_split = DatasetNumpy_MTS_XOR( n_epochs=2, batch_size=args.batch_size, nb_steps=args.nb_steps, channel_size=int(args.n_in/2), noise_rate=args.noise_rate )
        test_ds = DatasetNumpy_MTS_XOR( n_epochs=2, batch_size=args.batch_size, nb_steps=args.nb_steps, channel_size=int(args.n_in/2), noise_rate=args.noise_rate )

        train_loader_custom_collate = DataLoader(train_ds_split, args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader_custom_collate   = DataLoader(val_ds_split,   args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        test_loader_custom_collate  = DataLoader(test_ds,        args.batch_size, shuffle=None, collate_fn=custom_collate_fn)
        return train_loader_custom_collate, val_loader_custom_collate, test_loader_custom_collate
    
    # selecting the MNIST task
    elif args.dataset_name == 'mnist':
        # get the dataset
        if os.getcwd() == '/home/filippo/hsnn':
            root = '/home/filippo/data/audiospikes'
        elif os.getcwd() == '/Users/filippomoro/Documents/hsnn':
            root = '/Users/filippomoro/Documents/datasets'
        train_dataset = torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=True)
        test_dataset = torchvision.datasets.MNIST(root, train=False, transform=None, target_transform=None, download=True)
        
        # add the ficticious temporal dimention
        train_dataset.data  = expand_time_dim( train_dataset.data.reshape( train_dataset.data.shape[0], -1 )/255., nb_steps=args.nb_steps )
        test_dataset.data   = expand_time_dim( test_dataset.data.reshape( test_dataset.data.shape[0], -1 )/255., nb_steps=args.nb_steps )
        
        # split train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size   = len(train_dataset) - train_size
        train_ds_split, val_ds_split = random_split(train_dataset, [train_size, val_size])

        # produce the dataloader
        train_loader_custom_collate = DataLoader(train_ds_split, args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader_custom_collate   = DataLoader(val_ds_split,   args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        test_loader_custom_collate  = DataLoader(test_dataset,   args.batch_size, shuffle=None, collate_fn=custom_collate_fn)
        return train_loader_custom_collate, val_loader_custom_collate, test_loader_custom_collate

    elif args.dataset_name in ['s-mnist', 'ps-mnist']:
        # get the dataset
        if os.getcwd() == '/home/filippo/hsnn':
            root = '/home/filippo/data/audiospikes'
        elif os.getcwd() == '/Users/filippomoro/Documents/hsnn':
            root = '/Users/filippomoro/Documents/datasets'

        if args.dataset_name == 's-mnist':
            transform = torchvision.transforms.Compose(
             [torchvision.transforms.ToTensor(),
              torchvision.transforms.Lambda(lambda x: x.view(-1,1))
             ])
        elif args.dataset_name == 'ps-mnist':
            pixel_permutation = torch.randperm(28*28)
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x.view(-1,1)[pixel_permutation])
                ])
        else: transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                ])
            
        train_dataset = torchvision.datasets.MNIST(root, train=True, transform=transform, target_transform=None, download=True)
        test_dataset = torchvision.datasets.MNIST(root, train=False, transform=transform, target_transform=None, download=True)
        
        # split train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size   = len(train_dataset) - train_size
        train_ds_split, val_ds_split = random_split(train_dataset, [train_size, val_size])

        # produce the dataloader
        train_loader_custom_collate = DataLoader(train_ds_split, args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader_custom_collate   = DataLoader(val_ds_split,   args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        test_loader_custom_collate  = DataLoader(test_dataset,   args.batch_size, shuffle=None, collate_fn=custom_collate_fn)
        return train_loader_custom_collate, val_loader_custom_collate, test_loader_custom_collate

    else:
        print('Dataset name does not match any known taks. Please repeat the experiment with new dataset_name')
        return None, None, None