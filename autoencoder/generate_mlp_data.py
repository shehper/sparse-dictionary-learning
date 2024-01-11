"""
Generate training data for Sparse Autoencoder. 
"""
import os
import torch
import numpy as np
import time
import psutil
import sys

## Add the path to the transformer subdirectory as it contains model.py
sys.path.insert(0, '../transformer')
from model import GPTConfig, GPT

## define some parameters; these can be overwritten from command line
device = 'cpu'
seed = 1442
total_contexts = 400000 
contexts_per_batch = 500
tokens_per_context = 200 
convert_to_f16 = True # save activations in Half dtype instead of Float
dataset = 'openwebtext'
model_dir = 'out' # ignored if init_from is not 'resume'
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
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
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
data_dtype = torch.float16 if convert_to_f16 else torch.float32
sae_data = torch.zeros(total_contexts * tokens_per_context, n_ffwd, dtype=data_dtype)
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
    activations = model.get_gelu_acts(contexts).to(dtype=data_dtype) # (b, t, n_ffwd)
    
    # pick tokens_per_context (n) tokens from each context; and flatten the first two dimensions
    data = torch.stack([activations[i, torch.randint(block_size, (tokens_per_context,)), :] for i in range(contexts_per_batch)]).view(-1, activations.shape[-1]) #(b*n, n_ffwd)
    
    # store this data in sae_data tensor at locations chosen by shuffled_indices
    sae_data[shuffled_indices[batch * contexts_per_batch * tokens_per_context: (batch+1) * contexts_per_batch * tokens_per_context]] = data
    
    print(f"Computed activations for batch {batch}/{num_batches} in {(time.time()-start_time)/(batch+1):.2f} seconds;\
          memory available: {psutil.virtual_memory().available / (1024**3):.2f} GB; memory usage: {psutil.virtual_memory().percent}%")
   
## save sae_data in n_files files
os.makedirs('sae_data', exist_ok=True)
examples_per_file = total_contexts * tokens_per_context // n_files
for i in range(n_files):
    # notice .clone(); else torch.save would need the storage required for the whole tensor sae_data 
    # https://github.com/pytorch/pytorch/issues/1995
    torch.save(sae_data[i * examples_per_file: (i+1) * examples_per_file].clone(), f'sae_data/sae_data_{i}.pt')
    print(f'saved sae_data_{i}.pt in sae_data')