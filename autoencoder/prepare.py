"""
Generate training data for sparse autoencoder. 

Run on a Macbook as
python -u prepare.py --total_contexts=5000 --tokens_per_context=16 --dataset=shakespeare_char --gpt_dir=out-shakespeare-char

If constrained by node RAM, run this file multiple times to generate data.
Make sure to pass different seeds so as to avoid duplicates, e.g.
python prepare.py --seed=0
python prepare.py --seed=1
"""
import os
import torch
import numpy as np
import time
import psutil
from resource_loader import ResourceLoader

## define some parameters; these can be overwritten from command line
device = 'cpu'
seed = 0
total_contexts = int(2e6) # number of contexts to compute MLP activations on
contexts_per_batch = 500 
tokens_per_context = 200 # number of tokens from each context window to evaluate MLP activations on
dataset = 'openwebtext'
gpt_dir = 'out' # directory name inside ../transformer containing transformer model checkpoint
n_files = 20 # number of files in which data will be saved 

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# TODO: include a statement here that alerts you if a file named 'f{seed * n_files + i}.pt' already exists in the data directory.
# This is to avoid running this file twice with the same seed.

torch.manual_seed(seed)
resourceloader = ResourceLoader(
                            dataset=dataset, 
                            gpt_dir=gpt_dir,
                            )

# load tokenized text dataset 
text_data = resourceloader.load_text_data() 

## load GPT model weights -- we need it to compute MLP activations
gpt = resourceloader.load_transformer_model()

## retrieve block size and mlp dimension from model 
block_size = gpt.config.block_size
n_ffwd = 4 * gpt.config.n_embd 

## initiate tensor to store data 
# if this fails due to memory constraints, lower total_contexts or tokens_per_context
data_storage = torch.zeros(total_contexts * tokens_per_context, n_ffwd, dtype=torch.float32)
shuffled_indices = torch.randperm(total_contexts * tokens_per_context)

## compute MLP activations
start_time = time.time()
n_batches = total_contexts // contexts_per_batch
for batch in range(n_batches):

    x, _ = resourceloader.get_text_batch(contexts_per_batch)
    
    # compute MLP activations from the loaded model
    _, _ = gpt(x)
    activations = gpt.mlp_activation_hooks[0] # (b, t, n_ffwd)
    gpt.clear_mlp_activation_hooks() # free up memory
    
    # TODO: clean up the next 5 lines of code. They are hard to read

    # pick tokens_per_context (n) tokens from each context; and flatten the first two dimensions
    token_locs = torch.randint(block_size, (tokens_per_context,)) # (b , t)
    data = torch.stack([activations[i, token_locs, :] for i in range(contexts_per_batch)]).view(-1, activations.shape[-1]) #(b*n, n_ffwd)
    
    # store this data in sae_data tensor at locations chosen by shuffled_indices
    data_storage[shuffled_indices[batch * contexts_per_batch * tokens_per_context: (batch+1) * contexts_per_batch * tokens_per_context]] = data
    
    print(f"Computed activations for batch {batch}/{n_batches} in {(time.time()-start_time)/(batch+1):.2f} seconds;\
          memory available: {psutil.virtual_memory().available / (1024**3):.2f} GB; memory usage: {psutil.virtual_memory().percent}%")
   
## now save the data in n_files files.
autoencoder_data_dir = os.path.join(os.path.abspath('.'), 'data', dataset, f"{n_ffwd}")
os.makedirs(autoencoder_data_dir, exist_ok=True)
examples_per_file = total_contexts * tokens_per_context // n_files
for i in range(n_files):
    # notice .clone(); else torch.save would need the storage required for the whole tensor sae_data 
    # https://github.com/pytorch/pytorch/issues/1995
    torch.save(data_storage[i * examples_per_file: (i+1) * examples_per_file].clone(), f'{autoencoder_data_dir}/{seed * n_files + i}.pt')
    print(f'{seed * n_files + i}.pt in {autoencoder_data_dir}')