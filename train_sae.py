"""
Train a Sparse AutoEncoder model

Run on a macbook on a Shakespeare dataset as 
python train_sae.py --dataset=shakespeare_char --model_dir=out-shakespeare-char --eval_contexts=1000 --batch_size=128 --device=cpu --eval_interval=100
"""
import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
from model import GPTConfig, GPT
import numpy as np
import wandb
import time
import matplotlib.pyplot as plt
from PIL import Image
import io
import psutil

## hyperparameters
device = 'cuda'
seed = 1442
dataset = 'openwebtext'
model_dir = 'out' 
wandb_log = True
l1_coeff = 3e-3
learning_rate = 3e-4
gpt_batch_size = 16 # batch size for computing reconstruction nll 
batch_size = 8192 # 8192 for owt
n_features = 4096
eval_contexts = 10000 # 10 million in anthropic paper; but we can choose 1 million as OWT dataset is smaller
eval_context_tokens = 10 # same as anthropic paper
eval_interval = 500

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# variables that depend on input parameters
device_type = 'cuda' if 'cuda' in device else 'cpu'
eval_tokens = eval_contexts * eval_context_tokens
config['device_type'], config['eval_tokens'] = device_type, eval_tokens

## Define Autoencoder class, 
class AutoEncoder(nn.Module):
    def __init__(self, n, m, lam=0.003):
        # for us, n = d_MLP (a.k.a. n_ffwd) and m = number of features
        super().__init__()
        self.enc = nn.Linear(n, m)
        self.relu = nn.ReLU()
        self.dec = nn.Linear(m, n)
        self.lam = lam # coefficient of L_1 loss

    def forward(self, acts):
        # acts is of shape (b, n) where b = batch_size, n = d_MLP
        x = acts - self.dec.bias # (b, n)
        f = self.relu(self.enc(x)) # (b, m)
        reconst_acts = self.dec(f) # (b, n)
        mseloss = F.mse_loss(reconst_acts, acts) # scalar
        l1loss = F.l1_loss(f, torch.zeros(f.shape, device=f.device), reduction='sum') # scalar
        loss = mseloss + self.lam * l1loss # scalar
        return loss, f, reconst_acts, mseloss, l1loss

    # TODO: remove this method?    
    # @torch.no_grad()
    # def get_feature_acts(self, acts):
    #     # given acts of shape (b, n), compute feature activations
    #     x = acts - self.dec.bias # (b, n)
    #     f = self.relu(self.enc(x)) # (b, m)
    #     return f


# a function to remove components of gradients parallel to weights, needed during training
def remove_parallel_component(grad, weight):
    # remove gradient information parallel to weight vectors
    
    # compute projection of gradient onto weight
    # recall proj_b a = (a.\hat{b}) \hat{b} is the projection of a onto b

    unit_w = F.normalize(weight, dim=0) # \hat{b}
    proj = torch.sum(grad * unit_w, dim=0) * unit_w 

    return grad - proj


# a slightly modified version of nanoGPT get_batch function to get a batch of text data
def get_text_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y


## a helper function to convert a histogram to a PIL image so that it can be logged with wandb
def get_hist_image(data, bins='auto'):
    # plot a histogram
    _, ax = plt.subplots()
    ax.hist(data, bins=bins)
    ax.set_title('histogram')

    # save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # convert buffer to a PIL Image and return
    return Image.open(buf)


## a helper function that computes the number of tokens on which each feature is activated
# TODO: maybe this function needs to be removed
# def get_n_feature_acts():
    
#     # number of tokens on which a feature is active 
#     n_feature_activations = torch.zeros(n_features, dtype=torch.float32) # initiate with zeros

#     for iter in range(eval_contexts // gpt_batch_size):
        
#         if device_type == 'cuda':
#             # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
#             x = X[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].pin_memory().to(device, non_blocking=True)
#             y = Y[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].pin_memory().to(device, non_blocking=True)
#         else:
#             x = X[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].to(device)
#             y = Y[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].to(device)
        
#         mlp_activations = model.get_gelu_acts(x) # (gpt_b, t, n_ffwd)
#         feature_activations = sae.get_feature_acts(mlp_activations) # (b, t, n_features)
#         selected_feature_acts = torch.stack([feature_activations[i, selected_tokens_loc[iter], :] for i in range(gpt_batch_size)])  # (b, tokens_per_eval_context, n_features)
#         n_feature_activations += torch.count_nonzero(selected_feature_acts, dim=[0, 1]).to('cpu') # (n_features, )
    
#     return n_feature_activations



if __name__ == '__main__':
    
    torch.manual_seed(seed)
    # TODO: dont know if I need the following two lines, but leaving them here for now
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


    ## load tokenized text data
    data_dir = os.path.join('data', dataset)
    text_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')


    ## load GPT model
    ckpt_path = os.path.join(model_dir, 'ckpt.pt')
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
    block_size = model.config.block_size


    ## recall that mlp activations data was saved in the folder 'sae_data' in multiple files 
    n_parts = len(next(os.walk('sae_data'))[2]) # number of partitions of (or files in) sae_data
    
    # start here by loading the first partition
    n_part = 0 # partition number
    loading_start_time = time.time() # TODO: Can remove this later once I get a sense of the amount of time it takes
    curr_part = torch.load(f'sae_data/sae_data_{n_part}.pt') # current partition
    ex_per_part, n_ffwd = curr_part.shape # number of examples per partition, gpt d_mlp
    N = n_parts * ex_per_part # total number of training examples for autoencoder
    offset = 0 # when partition number > 0, first 'offset' # of examples will be trained with exs from previous partition
    print(f'successfully loaded the first partition of data from sae_data/sae_data_{n_part}.pt in {(time.time()-loading_start_time):.2f} seconds')
    print(f'Approximate number of training examples: {N}')

    memory = psutil.virtual_memory()
    print(f'Available memory after loading data: {memory.available / (1024**3):.2f} GB; memory usage: {memory.percent}%')

    ## Get text data for evaluation 
    X, Y = get_text_batch(text_data, block_size=block_size, batch_size=eval_contexts) # (eval_contexts, block_size) 
    # in each context, randomly select eval_context_tokens (=10 in Anthropic's paper) where 
    selected_tokens_loc = [torch.randint(block_size, (eval_context_tokens,)) for _ in range(eval_contexts)]
    # Note: for eval_contexts=1 million it will take 15.6GB of CPU MEMORY --- 7.81GB each for x and y
    # perhaps we will have to go one order of magnitude lower; use 0.1million contexts
    memory = psutil.virtual_memory()
    print(f'collected text data for evaluation; available memory: {memory.available / (1024**3):.2f} GB; memory usage: {memory.percent}%')


    ## Compute and store MLP activations, full transformer loss and ablated MLP loss on evaluation text data
    mlp_acts_storage = torch.tensor([], dtype=torch.float16)
    res_stream_storage = torch.tensor([], dtype=torch.float16)
    full_loss, mlp_ablated_loss = 0, 0
    for iter in range(eval_contexts // gpt_batch_size):    
        print(f'iter = {iter}/{eval_contexts // gpt_batch_size} in computation of mlp_acts and res_stream for eval data')
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x = X[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].pin_memory().to(device, non_blocking=True)
            y = Y[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].pin_memory().to(device, non_blocking=True)
        else:
            x = X[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].to(device)
            y = Y[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].to(device)
        res_stream, mlp_activations, batch_loss, batch_ablated_loss = model.forward_with_and_without_mlp(x, y)    
        mlp_acts_storage = torch.cat([mlp_acts_storage, mlp_activations.to(dtype=torch.float16, device='cpu')])
        res_stream_storage = torch.cat([res_stream_storage, res_stream.to(dtype=torch.float16, device='cpu')])
        full_loss += batch_loss
        mlp_ablated_loss += batch_ablated_loss
    full_loss /= (eval_contexts // gpt_batch_size)
    mlp_ablated_loss /= (eval_contexts // gpt_batch_size)
    memory = psutil.virtual_memory()
    print(f'computed mlp activations and losses on eval data; available memory: {memory.available / (1024**3):.2f} GB; memory usage: {memory.percent}%')
    print(f'The full transformer loss and MLP ablated loss on the evaluation data are {full_loss:.2f}, {mlp_ablated_loss:.2f}')
    # TODO: Do I really need X and Y anymore or can I delete them from the memory?


    ## initiate the autoencoder and optimizer
    sae = AutoEncoder(n_ffwd, n_features, lam=l1_coeff).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate) 

    
    ## WANDB LOG
    run_name = f'sae_{dataset}_{time.time():.0f}'
    if wandb_log:
        wandb.init(project=f'sae-{dataset}', name=run_name, config=config)


    ## TRAINING LOOP
    start_time = time.time()
    
    for step in range(N // batch_size):    
        
        ### ------ pick a batch of training examples ------ #####
        batch_start = step * batch_size - n_part * ex_per_part - offset
        batch_end = (step + 1) * batch_size - n_part * ex_per_part - offset
        batch = curr_part[batch_start: batch_end].to(torch.float32)
        
        # if reach the end of current partition, load the next partition into CPU memory
        if batch_end >= ex_per_part and n_part < n_parts - 1: 
            n_part += 1
            loading_start_time = time.time() # TODO: Can remove this later once I get a sense of the amount of time it takes
            del curr_part # free up memory
            curr_part = torch.load(f'sae_data/sae_data_{n_part}.pt')
            print(f"successfully loaded sae_data_{n_part}.pt in {(time.time()-loading_start_time):.2f} seconds")
            batch = torch.cat([batch, curr_part[:batch_size - len(batch)]]).to(torch.float32)
            offset = ex_per_part - batch_end
        
        assert len(batch) == batch_size, f"length of batch = {len(batch)} at step = {step} and partition number = {n_part} is not correct"
        
        if device_type == 'cuda':
            batch = batch.pin_memory().to(device, non_blocking=True)
        else:
            batch = batch.to(device)
        
        ## -------- forward pass, backward pass, remove gradient information parallel to decoder columns, optimizer step, ----- ##
        ## --------  normalize dictionary vectors ------- ##
        optimizer.zero_grad(set_to_none=True) 
        loss = sae(batch)[0]
        loss.backward()
        sae.dec.weight.grad = remove_parallel_component(sae.dec.weight.grad, sae.dec.weight)
        optimizer.step()
        sae.dec.weight = nn.Parameter(F.normalize(sae.dec.weight, dim=0))
        del batch

        
        ## log info
        if step % eval_interval == 0:
            # TODO: can evaluation and logging be done asynchronously? 
        
            start_logging_time = time.time()
            reconst_nll, sae_loss, sae_mse_loss, sae_l1loss, l0_norm = 0, 0, 0, 0, 0 
            n_feature_acts = torch.zeros(n_features, dtype=torch.float32) # initiate with zeros
            for iter in range(eval_contexts // gpt_batch_size):   
                if device_type == 'cuda':
                    batch_mlp_activations = mlp_acts_storage[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].pin_memory().to(device, non_blocking=True)
                    batch_res_stream = res_stream_storage[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].pin_memory().to(device, non_blocking=True)
                    y = Y[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].pin_memory().to(device, non_blocking=True)
                else:
                    batch_mlp_activations = mlp_acts_storage[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].to(device)
                    batch_res_stream = res_stream_storage[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].to(device)
                    y = Y[iter * gpt_batch_size: (iter + 1) * gpt_batch_size].to(device)

                with torch.no_grad():
                    batch_loss, batch_f, batch_reconst_acts, batch_mseloss, batch_l1loss = sae(batch_mlp_activations)

                del batch_mlp_activations

                # evaluate number of feature activations (number of tokens on which each feature activates)
                batch_f = batch_f.to('cpu')
                selected_feature_acts = torch.stack([batch_f[i, selected_tokens_loc[iter], :] for i in range(gpt_batch_size)])  # (b, tokens_per_eval_context, n_features)
                n_feature_acts += torch.count_nonzero(selected_feature_acts, dim=[0, 1]) # (n_features, )
                l0_norm += torch.sum(torch.count_nonzero(batch_f, dim=-1)).item()
                del batch_f, selected_feature_acts

                # Compute reconstructed loss from batch_reconst_acts
                batch_reconst_nll = model.loss_from_mlp_acts(batch_res_stream, batch_reconst_acts, y)
                reconst_nll += batch_reconst_nll
                sae_loss += batch_loss 
                sae_mse_loss += batch_mseloss
                sae_l1loss += batch_l1loss
                

                
            reconst_nll /= (eval_contexts // gpt_batch_size)
            sae_l1loss /= (eval_contexts // gpt_batch_size)
            sae_loss /= (eval_contexts // gpt_batch_size)
            sae_mse_loss /= (eval_contexts // gpt_batch_size)
            l0_norm /= (eval_contexts // gpt_batch_size)

            nll_score = (full_loss - reconst_nll)/(full_loss - mlp_ablated_loss)
            
            log_feature_density = np.log10(n_feature_acts[n_feature_acts != 0]/(eval_tokens)) # (n_features,)
            feat_density = get_hist_image(log_feature_density)

            min_log_feature_density = log_feature_density.min().item()
            num_alive_neurons = len(log_feature_density)
            
            del n_feature_acts, batch_reconst_acts, batch_res_stream, y
            del log_feature_density

            print(f"batch: {step}/{N // batch_size}, time per step: {(time.time()-start_time)/(step+1):.2f}, logging time = {(time.time()-start_logging_time):.2f}")
            
            if wandb_log:
                wandb.log({'losses/mse_loss': sae_mse_loss.item(),
                        'losses/l1_loss': sae_l1loss.item(),
                        'losses/total_loss': loss.item(),
                        'losses/nll_loss': reconst_nll,
                        'losses/nll_score': nll_score,
                        'debug/l0_norm': l0_norm,
                        'debug/mean_dictionary_vector_length': torch.mean(torch.linalg.vector_norm(sae.dec.weight, dim=0)),
                        "feature_density/feature_density_histograms": wandb.Image(feat_density),
                        "feature_density/min_log_feat_density": min_log_feature_density,
                        "feature_density/num_alive_neurons": num_alive_neurons,
                        })

    print(f'Exited loop after training on {N // batch_size * batch_size} examples')

    if wandb_log:
        wandb.finish()