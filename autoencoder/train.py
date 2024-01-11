"""
Train a Sparse AutoEncoder model

Run on a macbook on a Shakespeare dataset as 
python train_sae.py --dataset=shakespeare_char --gpt_dir=out-shakespeare-char --eval_contexts=1000 --eval_batch_size=16 --batch_size=128 --device=cpu --eval_interval=100 --n_features=1024 --save_checkpoint=False
"""
import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import io
import psutil
import gc
import sys
from autoencoder import AutoEncoder

## Add path to the transformer subdirectory as it contains GPT class in model.py
sys.path.insert(0, '../transformer')
from model import GPTConfig, GPT

## hyperparameters
device = 'cuda'
seed = 1442
dataset = 'openwebtext'
gpt_dir = 'out' 
wandb_log = True
l1_coeff = 3e-3
learning_rate = 3e-4
batch_size = 8192 # 8192 for owt
n_features = 4096
eval_contexts = 10000 # 10 million in anthropic paper; but we can choose 1e4 as OWT dataset is smaller
tokens_per_eval_context = 10 # same as anthropic paper
eval_interval = 1000 # number of training steps after which the autoencoder is evaluated
eval_batch_size = 16 # numnber of contexts in each text batch when evaluating the autoencoder model
save_checkpoint = True # whether to save model, optimizer, etc or not
save_interval = 10000 # number of training steps after which a checkpoint will be saved
out_dir = 'out_autoencoder' # directory containing trained autoencoder model weights
# and to be consistent with top_activations.py
resampling_interval = 25000 # number of training steps after which neuron resampling will be performed

def is_step_start_of_investigating_dead_neurons(step, x):
    """checks we should start investigating dead/alive neurons at this step.
    In Anthropic's paper, it is step # 12500, 37500, 62500 and 87500, i.e. an odd multiple of (resampling_interval//2)."""
    return (step > 0) and step % (x // 2) == 0 and (step // (x // 2)) % 2 != 0 and step < 4 * x

def is_step_in_the_phase_of_investigating_neurons(step, x):
    """checks if this step falls in a phase where we should be check for active neurons.
       In Anthropic's paper, it a step in the range [12500, 25000), [37500, 50000), [62500, 75000), or [87500, 100000). """
    milestones = [x, 2*x, 3*x, 4*x]
    for milestone in milestones:
        if milestone - x//2 <= step < milestone:
            return True
    return False

def get_text_batch(data, block_size, batch_size):
    """a simplfied version of nanoGPT get_batch function to get a batch of text data (simplification: it does NOT send the batch to a GPU)"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y

def load_data(step, batch_size, current_partition_index, current_partition, n_partitions, examples_per_partition, offset):
    """A custom data loader specific for our needs.  
    It assumes that data is stored in multiple files in the 'sae_data' folder. 
    It selects a batch from 'current_partition' sequentially. When 'current_partition' reaches its end, it loads the next partition. 
    Input:
        step: current training step
        batch_size: batch size
        current_partition_index: index of the current partition or file from sae_data folder
        current_partition: current partition or file from sae_data folder
        n_partitions: total number of files in sae_data
        examples_per_partition: number of examples in each partition
    Returns:
        batch: batch of data
        current_partition: different from the input current_partition if the input current_partition 
                            had less than batch_size examples that have not been previously selected
        current_partition_index: index of current_partition
        offset: number of examples of current_partition that 
    """
    batch_start = step * batch_size - current_partition_index * examples_per_partition - offset # index of the start of the batch in the current partition
    batch_end = (step + 1) * batch_size - current_partition_index * examples_per_partition - offset # index of the end of the batch in the current partition
    # check if the end of the batch is beyond the current partition
    if batch_end > examples_per_partition and current_partition_index < n_partitions - 1:
        # handle transition to next part
        remaining = examples_per_partition - batch_start
        batch = current_partition[batch_start:].to(torch.float32)
        current_partition_index += 1
        del current_partition; gc.collect()
        current_partition = torch.load(f'sae_data/sae_data_{current_partition_index}.pt')
        batch = torch.cat([batch, current_partition[:batch_size - remaining]]).to(torch.float32)
        offset = batch_size - remaining
    else:
        # normal batch processing
        batch = current_partition[batch_start:batch_end].to(torch.float32)
    assert len(batch) == batch_size, f"length of batch = {len(batch)} at step = {step} and partition number = {current_partition_index} is not correct"
    return batch, current_partition, current_partition_index, offset


def get_histogram_image(data, bins='auto'):
    """plots a histogram for data and converts it to a PIL image so that it can be logged with wandb"""
    _, ax = plt.subplots()
    ax.hist(data, bins=bins)
    ax.set_title('histogram')

    buf = io.BytesIO() # save the plot to a buffer
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return Image.open(buf) # convert buffer to a PIL Image and return

slice_fn = lambda storage: storage[iter * eval_batch_size: (iter + 1) * eval_batch_size] # slices a torch tensor; used in evaluation phase

if __name__ == '__main__':

    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    # variables that depend on input parameters
    config['device_type'] = device_type = 'cuda' if 'cuda' in device else 'cpu'
    config['eval_tokens'] = eval_tokens = eval_contexts * tokens_per_eval_context
        
    torch.manual_seed(seed)

    ## load tokenized text data
    current_dir = os.path.abspath('.')
    data_dir = os.path.join(os.path.dirname(current_dir), 'transformer', 'data', dataset)
    text_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

    ## load GPT model --- we need it to compute reconstruction nll and nll score
    ckpt_path = os.path.join(os.path.dirname(current_dir), 'transformer', gpt_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    gpt = GPT(gptconf)
    state_dict = checkpoint['model']
    compile = False # TODO: Don't know why I needed to set compile to False before loading the model..
    # TODO: Also, I dont know why the next 4 lines are needed. state_dict does not seem to have any keys with unwanted_prefix.
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

    ## LOAD THE FIRST FILE OF TRAINING DATA FOR AUTOENCODER 
    # recall that mlp activations data was saved in multiple files in the folder 'sae_data' 
    # we start by loading the first file here; other files are loaded inside load_data function as they are needed during training
    n_partitions = len(next(os.walk('sae_data'))[2]) # number of files in sae_data folder
    current_partition_index = 0 # partition number
    current_partition = torch.load(f'sae_data/sae_data_{current_partition_index}.pt') # current partition
    examples_per_partition, n_ffwd = current_partition.shape # number of examples per partition, gpt d_mlp
    total_training_examples = n_partitions * examples_per_partition # total number of training examples for autoencoder
    offset = 0 # when partition number > 0, first 'offset' 
    print(f'loaded the first partition of data from sae_data/sae_data_{current_partition_index}.pt')
    print(f'Approximate number of training examples: {total_training_examples}')

    memory = psutil.virtual_memory()
    print(f'Available memory after loading data: {memory.available / (1024**3):.2f} GB; memory usage: {memory.percent}%')

    ## LOAD DATA TO BE USED FOR NEURON RESAMPLING
    data_for_resampling_neurons = torch.load('data_for_resampling_neurons.pt')
    memory = psutil.virtual_memory()
    print(f'loaded data or resampling neurons, available memory now: {memory.available / (1024**3):.2f} GB; memory usage: {memory.percent}%')

    ## PREPARE FOR EVALUATION
    # We need text data during evaluation so that we can compute the reconstructed NLL loss and reconstructed NLL score
    # We pick a dataset of eval_contexts (=10000 by default) contexts, each of length block_size (=1024 by default)
    # We compute and save MLP activations and residual stream of the transformer model on these examples
    # We also compute the full transformer loss and MLP-ablated loss 
    # Finally, we pre-select indices of tokens_per_eval_context (=10 by default, as in Anthropic's paper) tokens in each context
    # and save it in token_indices. These will be used for the calculation of feature activation counts during evaluation 
    # TODO: There is probably no need to pre-select these indices. Perhaps remove token_indices and sample tokens during evaluation on the go?
    X, Y = get_text_batch(text_data, block_size=block_size, batch_size=eval_contexts) # (eval_contexts, block_size) 
    num_eval_batches = eval_contexts // eval_batch_size
    mlp_activations_storage = torch.tensor([], dtype=torch.float16)
    residual_stream_storage = torch.tensor([], dtype=torch.float16)
    transformer_loss, mlp_ablated_loss = 0, 0

    for iter in range(num_eval_batches):    
        print(f'iter = {iter}/{num_eval_batches} in computation of mlp_activations and residual_stream for evaluation data')
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = slice_fn(X).pin_memory().to(device, non_blocking=True) if device_type == 'cuda' else slice_fn(X).to(device) # select a batch of text data inputs
        y = slice_fn(Y).pin_memory().to(device, non_blocking=True) if device_type == 'cuda' else slice_fn(Y).to(device) # select a batch of text data outputs
        res_stream, mlp_activations, batch_loss, batch_ablated_loss = gpt.forward_with_and_without_mlp(x, y) # Transformer forward pass; compute residual stream, MLP activations and losses
        mlp_activations_storage = torch.cat([mlp_activations_storage, mlp_activations.to(dtype=torch.float16, device='cpu')]) # store MLP activations
        residual_stream_storage = torch.cat([residual_stream_storage, res_stream.to(dtype=torch.float16, device='cpu')]) # store residual stream
        transformer_loss, mlp_ablated_loss = transformer_loss + batch_loss, mlp_ablated_loss + batch_ablated_loss 
    transformer_loss, mlp_ablated_loss = transformer_loss/num_eval_batches, mlp_ablated_loss/num_eval_batches # divide by num_eval_batches to get mean values
    token_indices = torch.randint(block_size, (eval_contexts, tokens_per_eval_context)) # (eval_contexts, tokens_per_eval_context)

    memory = psutil.virtual_memory()
    print(f'computed mlp activations and losses on eval data; available memory: {memory.available / (1024**3):.2f} GB; memory usage: {memory.percent}%')
    print(f'The full transformer loss and MLP ablated loss on the evaluation data are {transformer_loss:.2f}, {mlp_ablated_loss:.2f}')
    del X; gc.collect() # will not need X anymore; instead res_stream_storage and mlp_acts_storage will be used


    ## INITIATE AUTOENCODER AND OPTIMIZER
    autoencoder = AutoEncoder(n_ffwd, n_features, lam=l1_coeff, resampling_interval=resampling_interval).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate) 
    
    ## prepare for logging and saving checkpoints
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
        batch, current_partition, current_partition_index, offset = load_data(step, batch_size, current_partition_index, current_partition, n_partitions, examples_per_partition, offset)
        batch = batch.pin_memory().to(device, non_blocking=True) if device_type == 'cuda' else batch.to(device)
        
        # forward, backward pass
        optimizer.zero_grad(set_to_none=True) 
        output = autoencoder(batch) # f has shape (batch_size, n_features) 
        output['loss'].backward()

        # remove component of gradient parallel to weight # TODO: Is there an alternative to this? how about some proximal policy type update?
        autoencoder.remove_parallel_component_of_decoder_gradient()
        
        # step
        optimizer.step()

        # periodically update the norm of dictionary vectors to make sure they don't deviate too far from 1.
        if step % 1000 == 0: 
            autoencoder.normalize_decoder_columns()
        
        # free up memory
        del batch; gc.collect(); torch.cuda.empty_cache() 

        ### ------------ perform neuron resampling ----------- ######
        # check if at this step, we should start investigating dead/alive neurons
        # in Anthropic's paper, this is done at step # 12500, 37500, 62500 and 87500, i.e. an odd multiple of (resampling_interval//2).
        if is_step_start_of_investigating_dead_neurons(step, resampling_interval):
            print(f'initiating investigation of dead neurons at step = {step}')
            autoencoder.initiate_dead_neurons()

        # check if this step falls in a phase where we should check for active neurons
        # in Anthropic's paper, this is a step in the range [12500, 25000), [37500, 50000), [62500, 75000), [87500, 100000). 
        if is_step_in_the_phase_of_investigating_neurons(step, resampling_interval):
            autoencoder.update_dead_neurons(output['f'])

        # if step is a multiple of resampling interval, perform neuron resampling
        if step > 0 and step % resampling_interval == 0:
            autoencoder.resample_neurons(data=data_for_resampling_neurons, optimizer=optimizer, batch_size=batch_size)

        # free up memory
        del output; gc.collect(); torch.cuda.empty_cache() 

        ### ------------ log info ----------- ######
        if (step % eval_interval == 0) or step == total_steps - 1:
            autoencoder.eval() 
            log_dict = {'losses/reconstructed_nll': 0, # log-likelihood loss using the output of autoencoder as MLP activations in the transformer model 
                        'losses/feature_activation_sparsity': 0, # L0-norm; average number of non-zero components of the feature activation vector f 
                        'losses/autoencoder_loss': 0, 'losses/mse_loss': 0, 'losses/l1_loss': 0, 
                        }
            feature_activation_counts = torch.zeros(n_features, dtype=torch.float32) # number of tokens on which each feature is active
            start_log_time = time.time()

            for iter in range(num_eval_batches):

                if device_type == 'cuda': # select batch of mlp activations, residual stream and y 
                    batch_mlp_activations = slice_fn(mlp_activations_storage).pin_memory().to(device, non_blocking=True) 
                    batch_res_stream = slice_fn(residual_stream_storage).pin_memory().to(device, non_blocking=True) 
                    batch_targets = slice_fn(Y).pin_memory().to(device, non_blocking=True) 
                else:
                    batch_mlp_activations = slice_fn(mlp_activations_storage).to(device)
                    batch_res_stream = slice_fn(residual_stream_storage).to(device)
                    batch_targets = slice_fn(Y).to(device)

                with torch.no_grad():
                    output = autoencoder(batch_mlp_activations) # output = {'loss': loss, 'f': f, 'reconst_acts': reconst_acts, 'mseloss': mseloss, 'l1loss': l1loss}
      
                f = output['f'].to('cpu') # (eval_batch_size, block_size, n_features)
                
                # restrict f to the subset of tokens_per_eval_context tokens determined by token_indices
                batch_token_indices = slice_fn(token_indices) # (eval_batch_size, tokens_per_eval_context)
                f_subset = torch.gather(f, 1, batch_token_indices.unsqueeze(-1).expand(-1, -1, n_features)) # (eval_batch_size, tokens_per_eval_context, n_features)  

                # for each feature, calculate the TOTAL number of tokens on which it is active; shape: (n_features, ) 
                feature_activation_counts += torch.count_nonzero(f_subset, dim=[0, 1]) # (n_features, )

                # calculat the AVERAGE number of non-zero entries in each feature vector
                log_dict['losses/feature_activation_sparsity'] += torch.mean(torch.count_nonzero(f_subset, dim=-1), dtype=torch.float32).item()
 
                del batch_mlp_activations, f, f_subset; gc.collect(); torch.cuda.empty_cache()

                # Compute reconstructed loss from batch_reconstructed_activations
                log_dict['losses/reconstructed_nll'] += gpt.loss_from_mlp_acts(batch_res_stream, output['reconst_acts'], batch_targets).item()
                log_dict['losses/autoencoder_loss'] += output['loss'].item() 
                log_dict['losses/mse_loss'] += output['mse_loss'].item()
                log_dict['losses/l1_loss'] += output['l1_loss'].item()
                del batch_res_stream, output, batch_targets; gc.collect(); torch.cuda.empty_cache()

            # take mean of all loss values by dividing by the number of evaluation batches
            log_dict = {key: val/num_eval_batches for key, val in log_dict.items()}

            # add nll score to log_dict
            log_dict['losses/nll_score'] = (transformer_loss - log_dict['losses/reconstructed_nll'])/(transformer_loss - mlp_ablated_loss).item()
            
            # compute feature densities and plot feature density histogram
            log_feature_activation_density = np.log10(feature_activation_counts[feature_activation_counts != 0]/(eval_tokens)) # (n_features,)
            feature_density_histogram = get_histogram_image(log_feature_activation_density)
            print(f"batch: {step}/{total_steps}, time per step: {(time.time()-start_time)/(step+1):.2f}, logging time = {(time.time()-start_log_time):.2f}")
            # print(log_dict)

            # log more metrics
            if wandb_log:
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
                wandb.log(log_dict)

            autoencoder.train()
                
        ### ------------ save a checkpoint ----------- ######
        if save_checkpoint and step > 0 and (step % save_interval == 0 or step == total_steps - 1):
            checkpoint = {
                    'autoencoder': autoencoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'log_dict': log_dict,
                    'config': config,
                    'feature_activation_counts': feature_activation_counts, # may be used later to identify alive vs dead neurons
                    }
            print(f"saving checkpoint to {out_dir}/{run_name} at training step = {step}")
            torch.save(checkpoint, os.path.join(out_dir, run_name, 'ckpt.pt'))


    if wandb_log:
        wandb.finish()    
    print(f'Finished training after training on {total_steps * batch_size} examples')



# TODO: Fix my training loops by including training on the last few examples with count < batch_size. As it is, I am ignoring them
# TODO: compile the gpt model and sae model for faster training?
# TODO: The function is_step_in_the_phase_of_investigating_neurons and the line 'if step and self.num_resamples < 4'
# assume that resampling will be done only 4 times. Perhaps I need to relax that?
# TODO: does the dtype of mlp activations affect the AutoEncoder performance?
# TODO: Can I somehow get more training data? Or train on repeated data? That might bring more neurons to life. 