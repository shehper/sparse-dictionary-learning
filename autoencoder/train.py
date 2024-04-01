"""
Train a Sparse AutoEncoder model

Run on a macbook on a Shakespeare dataset as 
python train.py --dataset=shakespeare_char --gpt_dir=out_sc_1_2_32 --eval_contexts=20 --eval_batch_size=16 --batch_size=128 --device=cpu --eval_interval=100 --n_features=1024 --resampling_interval=150 --wandb_log=True
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
import gc
from autoencoder import AutoEncoder
from resource_loader import ResourceLoader

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
eval_contexts = 10000 # number of windows of texts to evaluate the autoencoder on as OWT dataset is smaller
eval_interval = 1000 # number of training steps after which the autoencoder is evaluated
eval_batch_size = 16 # numnber of contexts in each text batch when evaluating the autoencoder model
save_checkpoint = True # whether to save model, optimizer, etc or not
save_interval = 10000 # number of training steps after which a checkpoint will be saved
out_dir = 'out' # directory containing trained autoencoder model weights
# and to be consistent with top_activations.py
resampling_interval = 25000 # number of training steps after which neuron resampling will be performed
num_resamples = 4 # number of times resampling is to be performed; it is done 4 times in Anthropic's paper

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

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# variables that depend on input parameters
config['device_type'] = device_type = 'cuda' if 'cuda' in device else 'cpu'
config['num_eval_batches'] = num_eval_batches = eval_contexts // eval_batch_size
    
torch.manual_seed(seed)
resourceloader = ResourceLoader(
                            dataset=dataset, 
                            gpt_dir=gpt_dir,
                            batch_size=batch_size,
                            device=device,
                            )

# load tokenized text dataset <-- openwebtext, shakespeare_char, etc
text_data = resourceloader.load_text_data() 

## load GPT model weights --- we need it to compute reconstruction nll and nll score
gpt = resourceloader.load_transformer_model()

## LOAD THE FIRST FILE OF TRAINING DATA FOR AUTOENCODER 
# autoencoder training data may have been saved in multiple files in the folder 'sae_data' # TODO: change folder name and path
# we start by loading the first file
autoencoder_data = resourceloader.load_autoencoder_data()

## load data for neuron resampling
# this data is used during neuron resampling procedure
resampling_data = resourceloader.load_resampling_data()

## INITIATE AUTOENCODER AND OPTIMIZER
# TODO: n and m should be called n_input, n_latents instead
autoencoder = AutoEncoder(n = 4 * resourceloader.transformer.config.n_embd, 
                            m = n_features, 
                            lam = l1_coeff, 
                            resampling_interval = resampling_interval).to(device)
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
num_steps = resourceloader.num_examples_total  // batch_size
# TODO: Update my training loop and load_data function to including the last few examples with count < batch_size? 
# As it is, I am ignoring them
for step in range(num_steps):

    # if step == 200:
    #     break

    ## load a batch of data    
    batch = resourceloader.get_autoencoder_data_batch(step)    
    if device_type == 'cuda':
        batch = batch.pin_memory().to(device, non_blocking=True)
    else:
        batch = batch.to(device)

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

    ## ------------ perform neuron resampling ----------- ######
    # check if at this step, we should start investigating dead/alive neurons
    # in Anthropic's paper, this is done at step # 12500, 37500, 62500 and 87500, i.e. an odd multiple of (resampling_interval//2).
    if autoencoder.is_step_start_of_investigating_dead_neurons(step, resampling_interval, num_resamples):
        print(f'initiating investigation of dead neurons at step = {step}')
        autoencoder.initiate_dead_neurons()

    # check if this step falls in a phase where we should check for active neurons
    # in Anthropic's paper, this is a step in the range [12500, 25000), [37500, 50000), [62500, 75000), [87500, 100000). 
    if autoencoder.is_step_in_the_phase_of_investigating_neurons(step, resampling_interval, num_resamples):
        autoencoder.update_dead_neurons(output['f'])

    # free up memory
    del batch, output; gc.collect(); torch.cuda.empty_cache() 
    
    # if step is a multiple of resampling interval, perform neuron resampling
    if (step+1) % resampling_interval == 0 and step < num_resamples * resampling_interval:
        print(f'{len(autoencoder.dead_neurons)} neurons to be resampled at step = {step}')
        autoencoder.resample_neurons(data=resampling_data, optimizer=optimizer, batch_size=batch_size)
    
    ### ------------ log info ----------- ######
    if (step % eval_interval == 0) or step == num_steps - 1:
        autoencoder.eval() 
        
        # a dictionary
        log_dict = {'losses/reconstructed_nll': 0, # log-likelihood loss using reconstructed MLP activations
                    'losses/l0_norm': 0, # L0-norm; average number of non-zero components of a feature activation vector
                    'losses/reconstruction_loss': 0, # |xhat - x|^2 <-- L2-norm between MLP activations & their reconstruction
                    'losses/l1_norm': 0, # L1-norm of feature activations
                    'losses/autoencoder_loss': 0, # reconstruction_loss + L1-coeff * l1_loss
                    'losses/nll_score': 0, # ratio of (nll_loss - ablated_loss) to (nll_loss - reconstructed_nll)
                    }
        
        # initiate a tensor, containing number of tokens on which each feature activates
        feat_acts_count = torch.zeros(n_features, dtype=torch.float32)

        # get batches of text data and evaluate the autoencoder on MLP activations
        for iter in range(num_eval_batches):
            x, y = resourceloader.get_text_batch(num_contexts=eval_batch_size)
            if device_type == 'cuda':
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True) 
            else:
                x, y = x.to(device), y.to(device)

            _, ablated_loss = gpt(x, y, mode="replace")
            _, nll_loss = gpt(x, y)
            mlp_acts = gpt.mlp_activation_hooks[0]
            gpt.clear_mlp_activation_hooks() # free up memory

            # forward pass on the autoencoder; get reconstructed nll
            with torch.no_grad():
                output = autoencoder(mlp_acts) # output = {'loss': loss, 'f': f, 'reconst_acts': reconst_acts, 'mseloss': mseloss, 'l1loss': l1loss}
            _, reconstructed_nll = gpt(x, y, mode="replace", replacement_tensor=output['reconst_acts'])

            # get feature activations from autoencoder output
            feat_acts = output['f'].to('cpu') # (eval_batch_size, block_size, n_features)
            
            # for each feature, calculate the TOTAL number of tokens on which it is active; shape: 
            torch.add(feat_acts_count, feat_acts.count_nonzero(dim=[0, 1]), out=feat_acts_count) # (n_features, )

            # calculat the AVERAGE number of non-zero entries in each feature vector and log all losses
            log_dict['losses/l0_norm'] += feat_acts.count_nonzero(dim=-1).float().mean().item()
            log_dict['losses/reconstructed_nll'] += reconstructed_nll
            log_dict['losses/autoencoder_loss'] += output['loss'].item() 
            log_dict['losses/reconstruction_loss'] += output['mse_loss'].item()
            log_dict['losses/l1_norm'] += output['l1_loss'].item()
            log_dict['losses/nll_score'] += (nll_loss - reconstructed_nll)/(nll_loss - ablated_loss).item()

        # compute feature densities and plot feature density histogram
        log_feat_acts_density = np.log10(feat_acts_count[feat_acts_count != 0]/(eval_contexts * gpt.config.block_size)) # (n_features,)
        feat_density_historgram = get_histogram_image(log_feat_acts_density)

        # take mean of all loss values by dividing by the number of evaluation batches
        log_dict = {key: val/num_eval_batches for key, val in log_dict.items()}
        
        # log more metrics
        log_dict.update(
                {'debug/mean_dictionary_vector_length': torch.linalg.vector_norm(autoencoder.dec.weight, dim=0).mean(),
                'feature_density/min_log_feat_density': log_feat_acts_density.min().item() if len(log_feat_acts_density) > 0 else -100,
                'feature_density/num_neurons_with_feature_density_above_1e-3': (log_feat_acts_density > -3).sum(),
                'feature_density/num_neurons_with_feature_density_below_1e-3': (log_feat_acts_density < -3).sum(),
                'feature_density/num_neurons_with_feature_density_below_1e-4': (log_feat_acts_density < -4).sum(), 
                'feature_density/num_neurons_with_feature_density_below_1e-5': (log_feat_acts_density < -5).sum(),
                'feature_density/num_alive_neurons': len(log_feat_acts_density),
                'training_step': step, 
                'training_examples': step * batch_size
                })
        if wandb_log:
            log_dict.update({'feature_density/feature_density_histograms': wandb.Image(feat_density_historgram)})    
            wandb.log(log_dict)

        autoencoder.train()
            
    ### ------------ save a checkpoint ----------- ######
    if save_checkpoint and step > 0 and (step % save_interval == 0 or step == num_steps - 1):
        checkpoint = {
                'autoencoder': autoencoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'log_dict': log_dict,
                'config': config,
                'feature_activation_counts': feat_acts_count, # may be used later to identify alive vs dead neurons
                }
        print(f"saving checkpoint to {out_dir}/{run_name} at training step = {step}")
        torch.save(checkpoint, os.path.join(out_dir, run_name, 'ckpt.pt'))

if wandb_log:
    wandb.finish()