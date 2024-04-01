"""
Generate training data for sparse autoencoder. 

Run on a Macbook as
python -u generate_mlp_data.py --total_contexts=5000 --tokens_per_context=16 --dataset=shakespeare_char --model_dir=out-shakespeare-char 
"""
import os
import torch
import numpy as np
import time
import psutil
import sys

## Add the path to the transformer subdirectory as it contains model.py
sys.path.insert(0, '../transformer')
from model import GPTConfig
from hooked_model import HookedGPT

## define some parameters; these can be overwritten from command line
device = 'cpu'
seed = 0
total_contexts = int(2e6) # should take 770 GB 
contexts_per_batch = 500
tokens_per_context = 200 
dataset = 'openwebtext'
model_dir = 'out' # directory name inside ../transformer that contains transformer model checkpoint
n_files = 20 # number of files in which data will be saved 

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

## load tokenized text data
current_dir = os.path.abspath('.')
data_dir = os.path.join(os.path.dirname(current_dir), 'transformer', 'data', dataset)
text_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

## load model
ckpt_path = os.path.join(os.path.dirname(current_dir), 'transformer', model_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
print(f'loaded transformer model checkpoint from {ckpt_path}')
gptconf = GPTConfig(**checkpoint['model_args'])
model = HookedGPT(gptconf)
state_dict = checkpoint['model']
compile = False # TODO: Don't know why I needed to set compile to False before loading the model..
# TODO: I dont know why the next 4 lines are needed. state_dict does not seem to have any keys with unwanted_prefix.
unwanted_prefix = '_orig_mod.' 
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

## retrieve block size and mlp dimension from model 
block_size = model.config.block_size
n_ffwd = 4 * model.config.n_embd

# Print memory info before initiating sae_data
memory = psutil.virtual_memory()
print(f"Total memory: {memory.total / (1024**3):.2f} GB")
print(f"Available memory before initiating sae_data: {memory.available / (1024**3):.2f} GB; memory usage: {memory.percent}%")

## initiate a torch tensor that will save data
sae_data = torch.zeros(total_contexts * tokens_per_context, n_ffwd)
shuffled_indices = torch.randperm(total_contexts * tokens_per_context)

# Print memory info after initiating sae_data; a large amount of memory must have been used for this
memory = psutil.virtual_memory()
print(f"Available memory after initiating sae_data: {memory.available / (1024**3):.2f} GB; memory usage: {memory.percent}%")

## compute activations in batches
start_time = time.time()
num_batches = total_contexts // contexts_per_batch
for batch in range(num_batches):
    # randomly pick contexts_per_batch starting points of contexts from the text data
    ix = torch.randint(len(text_data) - block_size, (contexts_per_batch,)) # (b, )
    
    # complete contexts with the chosen initial points
    contexts = torch.stack([torch.from_numpy((text_data[i:i+block_size]).astype(np.int64)) for i in ix]) # (b, t)
    
    # compute MLP activations from the loaded model
    _, _ = model(contexts)
    activations = model.mlp_activation_hooks[0] # (b, t, n_ffwd)
    model.clear_mlp_activation_hooks() # free up memory
    
    # pick tokens_per_context (n) tokens from each context; and flatten the first two dimensions
    data = torch.stack([activations[i, torch.randint(block_size, (tokens_per_context,)), :] for i in range(contexts_per_batch)]).view(-1, activations.shape[-1]) #(b*n, n_ffwd)
    
    # store this data in sae_data tensor at locations chosen by shuffled_indices
    sae_data[shuffled_indices[batch * contexts_per_batch * tokens_per_context: (batch+1) * contexts_per_batch * tokens_per_context]] = data
    
    print(f"Computed activations for batch {batch}/{num_batches} in {(time.time()-start_time)/(batch+1):.2f} seconds;\
          memory available: {psutil.virtual_memory().available / (1024**3):.2f} GB; memory usage: {psutil.virtual_memory().percent}%")
   
## save sae_data in n_files files
data_dir = os.path.join(current_dir, 'data', dataset, f"{n_ffwd}")
os.makedirs(data_dir, exist_ok=True)
examples_per_file = total_contexts * tokens_per_context // n_files

# now save the data in n_files files.
for i in range(n_files):
    # notice .clone(); else torch.save would need the storage required for the whole tensor sae_data 
    # https://github.com/pytorch/pytorch/issues/1995
    torch.save(sae_data[i * examples_per_file: (i+1) * examples_per_file].clone(), f'{data_dir}/{seed * n_files + i}.pt')
    print(f'{seed * n_files + i}.pt in {data_dir}')