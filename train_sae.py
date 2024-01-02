"""
Train a Sparse AutoEncoder model

Run on a macbook on a Shakespeare dataset as 
python train_sae.py --dataset=shakespeare_char --gpt_dir=out-shakespeare-char --eval_contexts=1000 --batch_size=128 --device=cpu --eval_interval=100 --n_features=1024 --save_checkpoint=False
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
import gc

## hyperparameters
device = 'cuda'
seed = 1442
dataset = 'openwebtext'
gpt_dir = 'out' 
wandb_log = True
l1_coeff = 3e-3
learning_rate = 3e-4
gpt_batch_size = 16 # batch size for computing reconstruction nll 
batch_size = 8192 # 8192 for owt
n_features = 4096
eval_contexts = 10000 # 10 million in anthropic paper; but we can choose 1 million as OWT dataset is smaller
eval_context_tokens = 10 # same as anthropic paper
eval_interval = 500
save_checkpoint = True
out_dir = 'out_autoencoder' # directory containing trained autoencoder model weights
resampling_interval = 25000 # perform neuron resampling after every 25000 steps as in Anthropic's paper

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# variables that depend on input parameters
device_type = 'cuda' if 'cuda' in device else 'cpu'
eval_tokens = eval_contexts * eval_context_tokens
config['device_type'], config['eval_tokens'] = device_type, eval_tokens

# TODO: Fix my training loops by including training on the last few examples with count < batch_size
# As it is, I am ignoring them

# TODO: replace gc.collect(); torch.cuda.empty_cache() by using a context manager of the form ManagedMemory 
# with an exit method that executes deletions and gc.collect(), torch.cuda_empty_cache. 
# This will clean up the code

# TODO: compile your gpt model and sae model for faster training?

# TODO: Priority list:
# Save and load the model
# implement neuron resampling
# manual inspection

# TODO: It seems that gpu memory is not being fully utilized so maybe I can increase gpt_batch_size. 

## Define Autoencoder class, 
class AutoEncoder(nn.Module):
    def __init__(self, n, m, lam=0.003):
        # for us, n = d_MLP (a.k.a. n_ffwd) and m = number of features
        # TODO: look at the initialization of this neural network
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

# a helper lambda to slice a torch tensor
slice_fn = lambda storage: storage[iter * gpt_batch_size: (iter + 1) * gpt_batch_size]

def load_data(step, batch_size, current_partition_index, current_partition, n_parts, examples_per_part, offset):
    
    batch_start = step * batch_size - current_partition_index * examples_per_part - offset # index of the start of the batch in the current partition
    batch_end = (step + 1) * batch_size - current_partition_index * examples_per_part - offset # index of the end of the batch in the current partition

    # Check if the end of the batch is beyond the current partition
    if batch_end > examples_per_part and current_partition_index < n_parts - 1:
        # Handle transition to next part
        remaining = examples_per_part - batch_start
        batch = current_partition[batch_start:].to(torch.float32)
        current_partition_index += 1
        del current_partition; gc.collect()
        current_partition = torch.load(f'sae_data/sae_data_{current_partition_index}.pt')
        batch = torch.cat([batch, current_partition[:batch_size - remaining]]).to(torch.float32)
        offset = batch_size - remaining
    else:
        # Normal batch processing
        batch = current_partition[batch_start:batch_end].to(torch.float32)
    
    assert len(batch) == batch_size, f"length of batch = {len(batch)} at step = {step} and partition number = {current_partition_index } is not correct"
    
    return batch, current_partition_index, current_partition, offset


if __name__ == '__main__':
    
    torch.manual_seed(seed)

    ## load tokenized text data
    data_dir = os.path.join('data', dataset)
    text_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

    ## load GPT model
    ckpt_path = os.path.join(gpt_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    gpt = GPT(gptconf)
    state_dict = checkpoint['model']
    compile = False # TODO: Don't know why I needed to set compile to False before loading the model..
    # TODO: I dont know why the next 4 lines are needed. state_dict does not seem to have any keys with unwanted_prefix.
    unwanted_prefix = '_orig_mod.' 
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    gpt.load_state_dict(state_dict)
    gpt.eval()
    gpt.to(device)
    if compile:
        gpt = torch.compile(gpt) # requires PyTorch 2.0 (optional)
    block_size = gpt.config.block_size

    ## LOAD AND SAVE DATA TO BE USED FOR NEURON RESAMPLING

    ## LOAD TRAINING DATA FOR AUTOENCODER 
    # recall that mlp activations data was saved in the folder 'sae_data' in multiple files 
    n_parts = len(next(os.walk('sae_data'))[2]) # number of partitions of (or files in) sae_data
    # start by loading the first partition
    current_partition_index = 0 # partition number
    current_partition = torch.load(f'sae_data/sae_data_{current_partition_index}.pt') # current partition
    examples_per_part, n_ffwd = current_partition.shape # number of examples per partition, gpt d_mlp
    total_training_examples = n_parts * examples_per_part # total number of training examples for autoencoder
    offset = 0 # when partition number > 0, first 'offset' # of examples will be trained with exs from previous partition
    print(f'loaded the first partition of data from sae_data/sae_data_{current_partition_index}.pt')
    print(f'Approximate number of training examples: {total_training_examples}')

    memory = psutil.virtual_memory()
    print(f'Available memory after loading data: {memory.available / (1024**3):.2f} GB; memory usage: {memory.percent}%')

    ## load data that will be used to resample neurons
    sae_data_for_resampling_neurons = torch.load('sae_data_for_resampling_neurons.pt')
    memory = psutil.virtual_memory()
    print(f'loaded data or resampling neurons, available memory now: {memory.available / (1024**3):.2f} GB; memory usage: {memory.percent}%')


    ## Get text data for evaluation 
    X, Y = get_text_batch(text_data, block_size=block_size, batch_size=eval_contexts) # (eval_contexts, block_size) 
    # in each context, randomly select eval_context_tokens (=10 in Anthropic's paper) where 
    selected_tokens_loc = [torch.randint(block_size, (eval_context_tokens,)) for _ in range(eval_contexts)]
    # Note: for eval_contexts=1 million it will take 15.6GB of CPU MEMORY --- 7.81GB each for x and y
    # perhaps we will have to go one order of magnitude lower; use 0.1million contexts
    memory = psutil.virtual_memory()
    print(f'collected text data for evaluation; available memory: {memory.available / (1024**3):.2f} GB; memory usage: {memory.percent}%')


    ## Compute and store MLP activations, full transformer loss and ablated MLP loss on evaluation text data
    mlp_activations_storage = torch.tensor([], dtype=torch.float16)
    residual_stream_storage = torch.tensor([], dtype=torch.float16)
    full_loss, mlp_ablated_loss = 0, 0
    num_eval_batches = eval_contexts // gpt_batch_size
    for iter in range(num_eval_batches):    
        
        print(f'iter = {iter}/{num_eval_batches} in computation of mlp_acts and res_stream for eval data')

        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = slice_fn(X).pin_memory().to(device, non_blocking=True) if device_type == 'cuda' else slice_fn(X).to(device)
        y = slice_fn(Y).pin_memory().to(device, non_blocking=True) if device_type == 'cuda' else slice_fn(Y).to(device)

        res_stream, mlp_activations, batch_loss, batch_ablated_loss = gpt.forward_with_and_without_mlp(x, y)    
        mlp_activations_storage = torch.cat([mlp_activations_storage, mlp_activations.to(dtype=torch.float16, device='cpu')])
        residual_stream_storage = torch.cat([residual_stream_storage, res_stream.to(dtype=torch.float16, device='cpu')])
        full_loss += batch_loss
        mlp_ablated_loss += batch_ablated_loss
    
    full_loss, mlp_ablated_loss = full_loss/num_eval_batches, mlp_ablated_loss/num_eval_batches
 
    memory = psutil.virtual_memory()
    print(f'computed mlp activations and losses on eval data; available memory: {memory.available / (1024**3):.2f} GB; memory usage: {memory.percent}%')
    print(f'The full transformer loss and MLP ablated loss on the evaluation data are {full_loss:.2f}, {mlp_ablated_loss:.2f}')
    del X; gc.collect() # will not need X anymore; instead res_stream_storage and mlp_acts_storage will be used


    ## initiate the autoencoder and optimizer
    autoencoder = AutoEncoder(n_ffwd, n_features, lam=l1_coeff).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate) 

    # normalize the decoder weights; TODO: this is on trial basis for now
    with torch.no_grad():
        autoencoder.dec.weight.data = F.normalize(autoencoder.dec.weight.data, dim=0)
    
    ## WANDB LOG
    run_name = f'autoencoder_{dataset}_{time.time():.0f}'
    if wandb_log:
        wandb.init(project=f'sparse-autoencoder-{dataset}', name=run_name, config=config)
    if save_checkpoint:
        os.makedirs(os.path.join(out_dir, run_name), exist_ok=True)


    ## TRAINING LOOP
    start_time = time.time()
    
    for step in range(total_training_examples // batch_size):
 
        ## load a batch of data        
        batch, current_partition_index, current_partition, offset = load_data(step, batch_size, current_partition_index, current_partition, n_parts, examples_per_part, offset)
        batch = batch.pin_memory().to(device, non_blocking=True) if device_type == 'cuda' else batch.to(device)
        
        ## training step
        optimizer.zero_grad(set_to_none=True) 
        loss, f, _, _, _ = autoencoder(batch) # f has shape (batch_size, n_features)
        loss.backward()
        # TODO: this business of removing parallel components seems fishy. What purpose does it serve if you have to normalize
        # the weights afterwards anyway? I hope the method I apply here works okay
        # remove gradient component paralle to weight
        autoencoder.dec.weight.grad = remove_parallel_component(autoencoder.dec.weight.grad, autoencoder.dec.weight)
        optimizer.step()
        # periodically update the norm of dictionary vectors
        if step % 1000 == 0: 
            with torch.no_grad():
                autoencoder.dec.weight.data = F.normalize(autoencoder.dec.weight.data, dim=0)
        del batch; gc.collect(); torch.cuda.empty_cache() 

        ## neuron resampling
        # start keeping track of dead neurons when step is an odd multiple of resampling_interval//2
        if step % (resampling_interval) == resampling_interval//2 and step < 1e5:
            dead_neurons = set([feature for feature in range(n_features)])

        # remove any autoencoder neurons from dead_neurons that are active in this training step
        if (resampling_interval//2) <= step < resampling_interval or (resampling_interval//2)*3 <= step < resampling_interval*2 \
        or (resampling_interval//2)*5 <= step < resampling_interval*3 or (resampling_interval//2)*7 <= step < resampling_interval*4:   
            # torch.count_nonzero(f, dim=0) counts the number of examples on which each feature is active
            # torch.count_nonzero(f, dim=0).nonzero() gives indices of alive features, which we discard from the set dead_neurons
            for feature_number in torch.count_nonzero(f, dim=0).nonzero().view(-1):
                dead_neurons.discard(feature_number.item())

        if (step + 1) % resampling_interval == 0 and step < 1e5:
            # compute the loss of the current model on 100 batches
            # choose a batch of data 
            # TODO: pick a batch of data
            #batch = sae_data_for_resampling_neurons[batch_start: ]
            raise NotImplemented



        ## log info
        if step % eval_interval == 0:
            
            start_logging_time = time.time()

            log_dict = {'losses/reconst_nll': 0, 'losses/autoencoder_loss': 0, 'losses/mse_loss': 0, 'losses/l1_loss': 0, 
                        'losses/feature_activation_sparsity': 0}
            feature_activation_counts = torch.zeros(n_features, dtype=torch.float32) # initiate with zeros

            for iter in range(num_eval_batches):

                # select batch of mlp activations, residual stream and y 
                if device_type == 'cuda':
                    batch_mlp_activations = slice_fn(mlp_activations_storage).pin_memory().to(device, non_blocking=True) 
                    batch_res_stream = slice_fn(residual_stream_storage).pin_memory().to(device, non_blocking=True) 
                    batch_targets = slice_fn(Y).pin_memory().to(device, non_blocking=True) 
                else:
                    batch_mlp_activations = slice_fn(mlp_activations_storage).to(device)
                    batch_res_stream = slice_fn(residual_stream_storage).to(device)
                    batch_targets = slice_fn(Y).to(device)

                with torch.no_grad():
                    batch_loss, batch_f, batch_reconst_acts, batch_mseloss, batch_l1loss = autoencoder(batch_mlp_activations)

                # evaluate number of feature activations (number of tokens on which each feature activates)
                batch_f = batch_f.to('cpu')
                selected_feature_acts = torch.stack([batch_f[i, selected_tokens_loc[iter], :] for i in range(gpt_batch_size)])  # (b, tokens_per_eval_context, n_features)
                feature_activation_counts += torch.count_nonzero(selected_feature_acts, dim=[0, 1]) # (n_features, )
                log_dict['losses/feature_activation_sparsity'] += torch.mean(torch.count_nonzero(batch_f.view(-1, batch_f.shape[-1]), dim=-1), dtype=torch.float16) 
                del batch_mlp_activations, batch_f, selected_feature_acts; gc.collect(); torch.cuda.empty_cache()

                # Compute reconstructed loss from batch_reconst_acts
                log_dict['losses/reconst_nll'] += gpt.loss_from_mlp_acts(batch_res_stream, batch_reconst_acts, batch_targets)
                log_dict['losses/autoencoder_loss'] += batch_loss 
                log_dict['losses/mse_loss'] += batch_mseloss
                log_dict['losses/l1_loss'] += batch_l1loss
                del batch_res_stream, batch_reconst_acts, batch_targets; gc.collect(); torch.cuda.empty_cache()

            # take mean of all loss values by dividing by number of evaluation batches
            log_dict = {key: val/num_eval_batches for key, val in log_dict.items()}

            # add nll score to log_dict
            log_dict['losses/nll_score'] = (full_loss - log_dict['losses/reconst_nll'])/(full_loss - mlp_ablated_loss)
            
            # compute feature density and plot a histogram
            log_feature_activation_density = np.log10(feature_activation_counts[feature_activation_counts != 0]/(eval_tokens)) # (n_features,)
            feature_density_histogram = get_hist_image(log_feature_activation_density)
            print(f"batch: {step}/{total_training_examples // batch_size}, time per step: {(time.time()-start_time)/(step+1):.2f}, logging time = {(time.time()-start_logging_time):.2f}")

            # log more metrics
            log_dict.update(
                    {'debug/mean_dictionary_vector_length': torch.mean(torch.linalg.vector_norm(autoencoder.dec.weight, dim=0)),
                    'feature_density/feature_density_histograms': wandb.Image(feature_density_histogram),
                    'feature_density/min_log_feat_density': log_feature_activation_density.min().item() if len(log_feature_activation_density) > 0 else -100,
                    'feature_density/num_alive_neurons': len(log_feature_activation_density),
                    'training_step': step, 
                    'training_examples': step * batch_size
                    })

            if wandb_log:
                wandb.log(log_dict)
                
            # save a checkpoint
            if step > 0 and save_checkpoint:
                checkpoint = {
                        'autoencoder': autoencoder.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'log_dict': log_dict,
                        'config': config
                        }
                print(f"saving checkpoint to {out_dir}/{run_name}")
                torch.save(checkpoint, os.path.join(out_dir, run_name, 'ckpt.pt'))

    if wandb_log:
        wandb.finish()    
    print(f'Finished training after training on {total_training_examples // batch_size * batch_size} examples')