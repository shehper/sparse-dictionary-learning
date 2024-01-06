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
gpt_dir = 'out' # TODO: this should be changed to 'out_gpt' for clarity; we might also have to make this change in train_gpt.py/train.py
wandb_log = True
l1_coeff = 3e-3
learning_rate = 3e-4
gpt_batch_size = 16 # batch size for computing reconstruction nll 
batch_size = 8192 # 8192 for owt
n_features = 4096
eval_contexts = 10000 # 10 million in anthropic paper; but we can choose 1 million as OWT dataset is smaller
eval_tokens_per_context = 10 # same as anthropic paper
eval_interval = 1000 # number of training steps after which the autoencoder is evaluated
save_checkpoint = True # whether to save model, optimizer, etc or not
save_interval = 10000 # number of training steps after which a checkpoint will be saved
out_dir = 'out_autoencoder' # directory containing trained autoencoder model weights # TODO: this name should be changed to autoencoder_dir for clarity 
# and to be consistent with top_activations.py
resampling_interval = 25000 # number of training steps after which neuron resampling will be performed

# TODO: change eval_tokens_per_context to tokens_per_eval_context for more clarity

# TODO: Do I need to specify subsets of tokens in neuron resampling? As it is, I don't think I have done that in data_for_resampling.py

# TODO: subsets_of_tokens_locations should probably just be called token_indices

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

# TODO: In recent experiments, feature activation sparsity starts off from infinity and decays fast to ~0. What's going on?

# TODO: how are neural networks initialized 

# TODO: log crude measures like count of neurons with feaeture density < 1e-4 and < 1e-5.

# TODO: I dont like the name gpt_batch_size. Change that. 

# TODO: log gpu metrics more closely 

# TODO: whats the intuition behind the nll score?

# TODO: Should I move the AutoEncoder class and all these helper functions to a different file?

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
    
    @torch.no_grad()
    def normalize_decoder_columns(self):
        autoencoder.dec.weight.data = F.normalize(autoencoder.dec.weight.data, dim=0)

    def remove_parallel_component_of_decoder_gradient(self):
        # remove gradient information parallel to weight vectors
        # to do so, compute projection of gradient onto weight
        # recall projection of a onto b is proj_b a = (a.\hat{b}) \hat{b}
        # here, a = grad, b = weight
        # TODO: check again that this function is correct, also rewrite it if it can be written in a simpler way
        unit_w = F.normalize(autoencoder.dec.weight, dim=0) # \hat{b}
        proj = torch.sum(autoencoder.dec.weight.grad * unit_w, dim=0) * unit_w 
        autoencoder.dec.weight.grad = autoencoder.dec.weight.grad - proj


# a slightly modified version of nanoGPT get_batch function to get a batch of text data
def get_text_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y


## a helper function to convert a histogram to a PIL image so that it can be logged with wandb
def get_histogram_image(data, bins='auto'):
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


def top_activations_and_tokens(f_subset, token_indices, contexts, context_on_each_side, k):
    """input: f_subset of shape (gpt_batch_size, eval_tokens_per_context, n_features)
              token_indices of shape  (gpt_batch_size, eval_tokens_per_context))
              contexts of shape (gpt_batch_size, block_size)
              context_on_each_side: an int, must satisfy (2 * context_on_each_side + 1) < block_size
              k: an int, the number of top activation values to compute and return, must be < gpt_batch_size * eval_tokens_per_context
    returns: 
              top_values: a tensor of shape (k, n_features)
              top_tokens_with_context: a tensor of shape (k, n_features, 2 * context_on_each_side + 1)"""

    #assert f.shape etc
    #assert token_indices.shape etc
    flattened_f = f_subset.view(-1, f_subset.shape[-1]) # (m * n, p)
    top_values, flattened_indices = torch.topk(flattened_f, k=k, dim=0) # (k, p) 
    unflattened_indices = torch.stack([flattened_indices // f_subset.shape[1], flattened_indices % f_subset.shape[1]], dim=2) # (k, p, 2)

    # new_indices contain locations of tokens with highest feature activations in the original contexts
    indices = unflattened_indices.clone()
    indices[:, :, 1] = token_indices[unflattened_indices[:, :, 0], unflattened_indices[:, :, 1]]

    top_tokens_with_contexts = torch.stack([contexts[indices[:, :, 0], indices[:, :, 1]+i] for i in range(-context_on_each_side, context_on_each_side + 1)], dim=2)

    return top_values, top_tokens_with_contexts


if __name__ == '__main__':

    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    # variables that depend on input parameters
    config['device_type'] = device_type = 'cuda' if 'cuda' in device else 'cpu'
    config['eval_tokens'] = eval_tokens = eval_contexts * eval_tokens_per_context
        
    torch.manual_seed(seed)

    ## load tokenized text data
    data_dir = os.path.join('data', dataset)
    text_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

    ## load GPT model --- we need it to compute reconstruction nll and nll score
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
    config['block_size'] = block_size = gpt.config.block_size

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

    ## LOAD DATA TO BE USED FOR NEURON RESAMPLING
    sae_data_for_resampling_neurons = torch.load('sae_data_for_resampling_neurons.pt')
    memory = psutil.virtual_memory()
    print(f'loaded data or resampling neurons, available memory now: {memory.available / (1024**3):.2f} GB; memory usage: {memory.percent}%')


    ## Get text data for evaluation 
    X, Y = get_text_batch(text_data, block_size=block_size, batch_size=eval_contexts) # (eval_contexts, block_size) 
    # in each context, randomly select eval_tokens_per_context (=10 in Anthropic's paper) where 
    token_indices = torch.randint(block_size, (eval_contexts, eval_tokens_per_context)) # (eval_contexts, tokens_per_eval_context)
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
    # TODO: when you do this, do you change the distribution from which the weights are drawn?
    # Perhaps computing weight.mean() and weight.std() will be useful. 
    # inituitively, normalization should not change mean but it could change std
    autoencoder.normalize_decoder_columns()
    
    ## WANDB LOG
    run_name = f'{time.time():.2f}-autoencoder-{dataset}'
    if wandb_log:
        import wandb
        wandb.init(project=f'sparse-autoencoder-{dataset}', name=run_name, config=config)
    if save_checkpoint:
        os.makedirs(os.path.join(out_dir, run_name), exist_ok=True)

    ## TRAINING LOOP
    start_time = time.time()
    total_steps = total_training_examples // batch_size

    for step in range(total_steps):
 
        ## load a batch of data        
        batch, current_partition_index, current_partition, offset = load_data(step, batch_size, current_partition_index, current_partition, n_parts, examples_per_part, offset)
        batch = batch.pin_memory().to(device, non_blocking=True) if device_type == 'cuda' else batch.to(device)
        
        # forward, backward pass
        optimizer.zero_grad(set_to_none=True) 
        loss, f, _, _, _ = autoencoder(batch) # f has shape (batch_size, n_features)
        loss.backward()

        # remove component of gradient parallel to weight # TODO: I am still not convinced this is the best approach
        autoencoder.remove_parallel_component_of_decoder_gradient()
        
        # step
        optimizer.step()

        # periodically update the norm of dictionary vectors to make sure they don't deviate too far from 1.
        if step % 1000 == 0: 
            autoencoder.normalize_decoder_columns()
        
        del batch; gc.collect(); torch.cuda.empty_cache() 


        ## log info
        if step % eval_interval == 0:
            
            start_logging_time = time.time()

            log_dict = {'losses/reconstructed_nll': 0, 'losses/autoencoder_loss': 0, 'losses/mse_loss': 0, 'losses/l1_loss': 0, 
                        'losses/feature_activation_sparsity': 0}
            feature_activation_counts = torch.zeros(n_features, dtype=torch.float32) # initiate with zeros

            # key: feature number; val: list of tuples (top activation value, corresponding token) 
            tokens_with_top_feature_activations = {i: [] for i in range(n_features)}

            # SIMPLEST LOGIC:
            # in batch_f_on_subset_of_tokens, find 10 values of (context, eval_token_in_context, activation_value) for each feature
            # use context, eval_token_in_context, batch_targets and subset_of_tokens to find the token id that gave the highest activation value
            # add (top_activation_value, token) to the list for each feature in the dictionary above
            # sort each list and keep only top 10 values
            # 

            # perhaps later I can find a faster solution later but this should work in the first go. 

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
                    batch_loss, batch_f, batch_reconstructed_activations, batch_mseloss, batch_l1loss = autoencoder(batch_mlp_activations)
                    
                batch_f = batch_f.to('cpu') # (gpt_batch_size, block_size, n_features)
                batch_token_indices = slice_fn(token_indices) # (gpt_batch_size, eval_tokens_per_context)

                # restrict batch_f to a subset of eval_tokens_per_context tokens in each context; shape: (gpt_batch_size, eval_tokens_per_context, n_features)
                batch_f_subset = torch.gather(batch_f, 1, batch_token_indices.unsqueeze(-1).expand(-1, -1, n_features))  

                # for each feature, calculate the TOTAL number of tokens on which it is active; shape: (n_features, ) 
                feature_activation_counts += torch.count_nonzero(batch_f_subset, dim=[0, 1]) # (n_features, )

                # calculat the AVERAGE number of non-zero entries in each feature vector
                log_dict['losses/feature_activation_sparsity'] += torch.mean(torch.count_nonzero(batch_f_subset, dim=-1), dtype=torch.float32)
 
                del batch_mlp_activations, batch_f, batch_f_subset; gc.collect(); torch.cuda.empty_cache()

                # Compute reconstructed loss from batch_reconstructed_activations
                log_dict['losses/reconstructed_nll'] += gpt.loss_from_mlp_acts(batch_res_stream, batch_reconstructed_activations, batch_targets)
                log_dict['losses/autoencoder_loss'] += batch_loss 
                log_dict['losses/mse_loss'] += batch_mseloss
                log_dict['losses/l1_loss'] += batch_l1loss
                del batch_res_stream, batch_reconstructed_activations, batch_targets; gc.collect(); torch.cuda.empty_cache()

            # take mean of all loss values by dividing by number of evaluation batches
            log_dict = {key: val/num_eval_batches for key, val in log_dict.items()}

            # add nll score to log_dict
            log_dict['losses/nll_score'] = (full_loss - log_dict['losses/reconstructed_nll'])/(full_loss - mlp_ablated_loss)
            
            # compute feature density and plot a histogram
            log_feature_activation_density = np.log10(feature_activation_counts[feature_activation_counts != 0]/(eval_tokens)) # (n_features,)
            feature_density_histogram = get_histogram_image(log_feature_activation_density)
            print(f"batch: {step}/{total_steps}, time per step: {(time.time()-start_time)/(step+1):.2f}, logging time = {(time.time()-start_logging_time):.2f}")

            # log more metrics
            log_dict.update(
                    {'debug/mean_dictionary_vector_length': torch.mean(torch.linalg.vector_norm(autoencoder.dec.weight, dim=0)),
                    'feature_density/feature_density_histograms': wandb.Image(feature_density_histogram),
                    'feature_density/min_log_feat_density': log_feature_activation_density.min().item() if len(log_feature_activation_density) > 0 else -100,
                    'feature_density/num_neurons_with_feature_density_above_1e-3': (log_feature_activation_density > -3).sum(),
                    'feature_density/num_neurons_with_feature_density_below_1e-3': (log_feature_activation_density < -3).sum(),
                    'feature_density/num_neurons_with_feature_density_below_1e-4': (log_feature_activation_density < -4).sum(), 
                    'feature_density/num_neurons_with_feature_density_below_1e-5': (log_feature_activation_density < -5).sum(),
                    'feature_density/num_alive_neurons': len(log_feature_activation_density),
                    'training_step': step, 
                    'training_examples': step * batch_size
                    })

            if wandb_log:
                wandb.log(log_dict)
                
            # save a checkpoint 
            # TODO: perhaps I should save a checkpoint BEFORE resampling neurons
        if step > 0 and (step % save_interval == 0 or step == total_steps - 1) and save_checkpoint:
            checkpoint = {
                    'autoencoder': autoencoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'log_dict': log_dict,
                    'config': config,
                    'feature_activation_counts': feature_activation_counts, # may be used later to identify alive vs dead neurons
                    }
            print(f"saving checkpoint to {out_dir}/{run_name} at training step = {step}")
            torch.save(checkpoint, os.path.join(out_dir, run_name, 'ckpt.pt'))

        # TODO: modify this code somehow to allow for neuron resampling more than 4 times and at step > 1e5 perhaps?
        # or maybe just impose a condition that neurons are resampled only 4 times in total

        # TODO: neuron resampling should be a method of AutoEncoder and the code below should be simplified to 2-3 lines

        ## neuron resampling
        # start keeping track of dead neurons when step is an odd multiple of resampling_interval//2
        if step % (resampling_interval) == resampling_interval//2 and step < 1e5:
            dead_neurons = set([feature for feature in range(n_features)])
            initial_step_for_neuron_tracking = step # the step at which we started tracking dead neurons 
            
        # remove any autoencoder neurons from dead_neurons that are active in this training step
        if (resampling_interval//2) <= step < resampling_interval or (resampling_interval//2)*3 <= step < resampling_interval*2 \
        or (resampling_interval//2)*5 <= step < resampling_interval*3 or (resampling_interval//2)*7 <= step < resampling_interval*4:   
            # torch.count_nonzero(f, dim=0) counts the number of examples on which each feature is active
            # torch.count_nonzero(f, dim=0).nonzero() gives indices of alive features, which we discard from the set dead_neurons
            for feature_number in torch.count_nonzero(f, dim=0).nonzero().view(-1):
                dead_neurons.discard(feature_number.item())

            if step % 100 == 0:
                print(f'At training step = {step}, there are {len(dead_neurons)} neurons that have not fired since training step = {initial_step_for_neuron_tracking}')


        if (step + 1) % resampling_interval == 0 and step < 1e5:
            # compute the loss of the current model on 100 batches
            # choose a batch of data 
            # TODO: pick a batch of data
            #batch = sae_data_for_resampling_neurons[batch_start: ] 
            if len(dead_neurons) > 0:           
                temp_encoder_layer = nn.Linear(n_ffwd, n_features, device=device)
                temp_decoder_layer = nn.Linear(n_features, n_ffwd, device=device)
                with torch.no_grad():
                    autoencoder.enc.weight[torch.tensor(list(dead_neurons))] = temp_encoder_layer.weight[torch.tensor(list(dead_neurons))]
                    autoencoder.dec.weight[:, torch.tensor(list(dead_neurons))] = temp_decoder_layer.weight[:, torch.tensor(list(dead_neurons))]
                del temp_encoder_layer, temp_decoder_layer; gc.collect(); torch.cuda.empty_cache()
                print(f'resampled {len(dead_neurons)} neurons at training step = {step}')

    if wandb_log:
        wandb.finish()    
    print(f'Finished training after training on {total_steps * batch_size} examples')