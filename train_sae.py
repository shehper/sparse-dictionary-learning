"""
Train a Sparse AutoEncoder on MLP activations
"""

# TODO: compute mlp activations on gpu, send them to a buffer on cpu, 
# NEXT: while training the autoencoder, move a batch of activations back to the gpu
# And so son

# TODO: Check that the default initialization is the one mentioned in the paper. 

import os
from contextlib import nullcontext
import torch
import torch.nn as nn 
import torch.nn.functional as F
from model import GPTConfig, GPT
import numpy as np
import wandb
import time

# load transformer training dataset and define get_batch
dataset = 'shakespeare_char'
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
device = 'cuda'
device_type = 'cuda' if 'cuda' in device else 'cpu'

gpt_dir = 'out-shakespeare-char' # ignored if init_from is not 'resume'

wandb_log = True
batch_size = 10
n_steps = 3600
block_size = 12 # length of context window
n_tokens = block_size//4 # number of tokens from each context
buffer_contexts = 50 * batch_size # number of contexts in buffer
n_features = 1024 # change this to 4096 for owt

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

### -------- refill_interval depends on buffer_contexts
# let it be an even multiple of batch size so that after an integer number of steps, buffer is exactly half-used
assert buffer_contexts % (2*batch_size) == 0, "adjust buffer_contexts so that it is an even multiple of batch_size"
# There are buffer_contexts * n_tokens activations in the buffer
refill_interval = int(buffer_contexts * n_tokens/(2*batch_size))

## load the pre-trained transformer model 
ckpt_path = os.path.join(gpt_dir, 'ckpt.pt')
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

## get contexts
seed = 1337
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def initial_data(b, seed=0, n=256, t=1024, train_data=train_data):
    # get b contexts, n < t tokens 
    # returns b*n activation vectors
    assert n <= t, "Number of tokens chosen must not exceed context window length"

    torch.manual_seed(seed)
    ix = torch.randint(len(train_data) - block_size, (b,))
    contexts = torch.stack([torch.from_numpy((train_data[i:i+block_size]).astype(np.int64)) for i in ix]) # (b, t)

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        contexts = contexts.pin_memory().to(device, non_blocking=True)
    else:
        contexts = contexts.to(device)

    activations = model.get_gelu_acts(contexts) # (b, t, n_ffwd)
    
    # sample n tokens from each context and flatten the batch and token dimension
    data = torch.stack([activations[i, torch.randint(t, (n,)), :] for i in range(b)]).view(-1, activations.shape[-1]) #(b*n, n_ffwd)

    # randomly shuffle all activation vectors and return
    return data[torch.randperm(b*n)] 

def refill_data(data, seed=0, b=100, n=256, t=1024):
    # remove the first N//2 contexts as they have already been used 
    # fill new contexts and shuffle again
    torch.manual_seed(seed)
    N, n_ffwd = data.shape # N = b*n/2
    assert N == b * n, "there is some issue with shape of data"
    data = data[N//2:] # remove the first half of activation vectors 
    ix = torch.randint(len(train_data) - block_size, (b//2,)) # pick new b//2 contexts
    contexts = torch.stack([torch.from_numpy((train_data[i:i+block_size]).astype(np.int64)) for i in ix]) # (b//2, t)
    activations = model.get_gelu_acts(contexts) # (b//2, t, n_ffwd)

    # sample n tokens from each context and flatten the batch and token dimension
    new_data = torch.stack([activations[i, torch.randint(t, (n,)), :] for i in range(b//2)]).view(-1, n_ffwd) # (b//2 * n, n_ffwd)
    data = torch.cat((data, new_data)) 
    return data[torch.randperm(n * b)]

class AutoEncoder(nn.Module):
    def __init__(self, n, m, lam=0.003):
        # for us, n will be d_MLP and m will be the number of features
        super().__init__()
        self.enc = nn.Linear(n, m)
        self.relu = nn.ReLU()
        self.dec = nn.Linear(m, n)
        self.lam = lam # coefficient of L_1 loss

    def forward(self, acts):
        # acts is of shape (.., n) where .. are batch dimensions
        x = acts - self.dec.bias # (.., n)
        f = self.relu(self.enc(x)) # (.., m)
        x = self.dec(f) # (.., n)
        mseloss = F.mse_loss(x, acts) # scalar
        l1loss = F.l1_loss(f, torch.zeros(f.shape), reduction='sum') # scalar
        loss = mseloss + self.lam * l1loss # scalar
        out = {'mse_loss': mseloss, 'l1loss': l1loss, 
                'loss': loss, 'recons_acts': x, 'f': f}
        return loss, out
    
def update_grad(grad, weight):
    # remove gradient information parallel to weight vectors
    
    # compute projection of gradient onto weight
    # recall proj_b a = (a.\hat{b}) \hat{b} is the projection of a onto b

    unit_w = F.normalize(weight, dim=0) # \hat{b}
    proj = torch.sum(grad * unit_w, dim=0) * unit_w 

    return grad - proj

def get_batch(split): # not modifying this function from nanoGPT train.py but will always just pass split='train'
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# ------------------------- #########

# load initial data
data = initial_data(b=buffer_contexts, n=n_tokens, t=block_size) 
print(f"data has shape {tuple(data.shape)}")

torch.manual_seed(0)
d_mlp = data.shape[-1] # MLP activation dimension
sae = AutoEncoder(d_mlp, n_features, lam=1e-3)
optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)
batch = 0

if wandb_log:
    wandb.init(project=f'sae-{dataset}', name=f'sae_{dataset}_{time.time()}')

for i in range(n_steps):    
    if i > 0 and i % refill_interval == 0:
        print(f'updating data buffer after {i} steps')
        data = refill_data(data, seed=i, b=buffer_contexts, n=n_tokens, t=block_size)
        batch = 0

    curr_batch = data[batch * batch_size: (batch + 1) * batch_size]
    loss, out = sae(curr_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # remove gradient information parallel to the decoder columns
    sae.dec.weight.grad = update_grad(sae.dec.weight.grad, sae.dec.weight)
    optimizer.step()

    # normalize decoder columns
    sae.dec.weight = nn.Parameter(F.normalize(sae.dec.weight, dim=0))

    batch += 1

    if i % 100 == 0:
        
        xs, ys = get_batch('train')
        reconstructed_nll_loss = model.reconstructed_loss(sae, xs, ys)
        
        print(f"batch: {i}/{n_steps}, mse loss: {out['mse_loss'].item():.2f}, l1_loss: {out['l1loss'].item():.2f}, \
            total_loss = {loss.item():.2f}, nll loss: {reconstructed_nll_loss:.2f}")

        if wandb_log:
            wandb.log({'losses/mse_loss': out['mse_loss'].item(),
                    'losses/l1_loss': out['l1loss'].item(),
                    'losses/total_loss': loss.item(),
                    'losses/nll_loss': reconstructed_nll_loss,
                    'debug/l0_norm': torch.mean(torch.count_nonzero(out['f'], dim=-1), dtype=torch.float32),
                    'debug/dictionary_vector_ave_length': torch.mean(torch.linalg.vector_norm(sae.dec.weight, dim=0)),
                    })
        
        #TODO: compute feature density histograms

    # if i > 0 and i % 25000 == 0:
    # TODO: resample neurons

if wandb_log:
    wandb.finish()