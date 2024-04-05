"""
Train a Sparse AutoEncoder model

Run on a macbook on a Shakespeare dataset as 
python train.py --dataset=shakespeare_char --gpt_ckpt_dir=out_sc_1_2_32 --eval_iters=1 --eval_batch_size=16 --batch_size=128 --device=cpu --eval_interval=100 --n_features=1024 --resampling_interval=150 --wandb_log=True
"""
import os
import torch
import numpy as np
import time
from autoencoder import AutoEncoder
from resource_loader import ResourceLoader
from utils.plotting_utils import make_histogram_image

## hyperparameters
# dataset and model
dataset = 'openwebtext'
gpt_ckpt_dir = 'out' 
# training
n_features = 4096
batch_size = 8192 # batch size for autoencoder training
l1_coeff = 3e-3
learning_rate = 3e-4
resampling_interval = 25000 # number of training steps after which neuron resampling will be performed
num_resamples = 4 # number of times resampling is to be performed; it is done 4 times in Anthropic's paper
resampling_data_size = 819200
# evaluation
eval_batch_size = 16 # batch size (number of GPT contexts) for evaluation
eval_iters = 200 # number of iterations in the evaluation loop
eval_interval = 1000 # number of training steps after which the autoencoder is evaluated
# I/O
save_checkpoint = True # whether to save model, optimizer, etc or not
save_interval = 10000 # number of training steps after which a checkpoint will be saved
out_dir = 'out' # directory containing trained autoencoder model weights
# wandb logging
wandb_log = True
# system
device = 'cuda'
# reproducibility
seed = 1442

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
    
torch.manual_seed(seed)
# initiating ResourceLoader in training mode loads Transformer checkpoint, text data, and autoencoder data
resourceloader = ResourceLoader(
                            dataset=dataset, 
                            gpt_ckpt_dir=gpt_ckpt_dir,
                            device=device,
                            mode="train",
                            )

gpt = resourceloader.transformer # TODO: either it should be called transformer or gpt
autoencoder = AutoEncoder(n_inputs = 4 * resourceloader.transformer.config.n_embd, 
                            n_latents = n_features, 
                            lam = l1_coeff, 
                            resampling_interval = resampling_interval).to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate) 

## prepare for logging and saving checkpoints
run_name = f'{time.time():.2f}'
if wandb_log:
    import wandb
    wandb.init(project=f'sparse-autoencoder-{dataset}', name=run_name, config=config)
if save_checkpoint:
    ckpt_path = os.path.join(out_dir, dataset, run_name)
    os.makedirs(ckpt_path, exist_ok=True)

############## TRAINING LOOP ###############
start_time = time.time()
num_steps = resourceloader.autoencoder_data_info["total_examples"]  // batch_size

for step in range(num_steps):
 
    batch = resourceloader.get_autoencoder_data_batch(step, batch_size=batch_size)
    optimizer.zero_grad(set_to_none=True) 
    autoencoder_output = autoencoder(batch) # f has shape (batch_size, n_features) 
    autoencoder_output['loss'].backward()

    # remove component of gradient parallel to weight
    autoencoder.remove_parallel_component_of_decoder_grad()
    optimizer.step()

    # periodically update the norm of dictionary vectors to ensure they stay close to 1.
    if step % 1000 == 0: 
        autoencoder.normalize_decoder_columns()

    ## ------------ perform neuron resampling ----------- ######
    # check if we should start investigating dead/alive neurons at this step
    # This is done at an odd multiple of resampling_interval // 2 in Anthropic's paper.
    if autoencoder.is_dead_neuron_investigation_step(step, resampling_interval, num_resamples):
        print(f'initiating investigation of dead neurons at step = {step}')
        autoencoder.initiate_dead_neurons()

    # check if we should look for dead neurons at this step
    # This is done between an odd and an even multiple of resampling_interval // 2.
    if autoencoder.is_within_neuron_investigation_phase(step, resampling_interval, num_resamples):
        autoencoder.update_dead_neurons(autoencoder_output['latents']) 
    
    # perform neuron resampling if step is a multiple of resampling interval
    if (step+1) % resampling_interval == 0 and step < num_resamples * resampling_interval:
        num_dead_neurons = len(autoencoder.dead_neurons)
        print(f'{num_dead_neurons} neurons to be resampled at step = {step}')
        if num_dead_neurons > 0:
            autoencoder.resample_dead_neurons(data=resourceloader.select_resampling_data(size=resampling_data_size), 
                                              optimizer=optimizer, 
                                              batch_size=batch_size)
    
    ### ------------ log info ----------- ######
    if (step % eval_interval == 0) or step == num_steps - 1:
        print(f'Entering evaluation mode at step = {step}')
        autoencoder.eval() 
        
        log_dict = {'losses/reconstructed_nll': 0, # log-likelihood loss using reconstructed MLP activations
                    'losses/l0_norm': 0, # L0-norm; average number of non-zero components of a feature activation vector
                    'losses/reconstruction_loss': 0, # |xhat - x|^2 <-- L2-norm between MLP activations & their reconstruction
                    'losses/l1_norm': 0, # L1-norm of feature activations
                    'losses/autoencoder_loss': 0, # reconstruction_loss + L1-coeff * l1_loss
                    'losses/nll_score': 0, # ratio of (nll_loss - ablated_loss) to (nll_loss - reconstructed_nll)
                    }
        
        # initiate a tensor containing the number of tokens on which each feature activates
        feat_acts_count = torch.zeros(n_features, dtype=torch.float32)

        # get batches of text data and evaluate the autoencoder on MLP activations
        for iter in range(eval_iters):
            if iter % 20 == 0:
                print(f'Performing evaluation at iterations # ({iter} - {min(iter+19, eval_iters)})/{eval_iters}')
            x, y = resourceloader.get_text_batch(num_contexts=eval_batch_size)

            _, nll_loss = gpt(x, y)
            mlp_acts = gpt.mlp_activation_hooks[0]
            gpt.clear_mlp_activation_hooks() # free up memory
            _, ablated_loss = gpt(x, y, mode="replace")

            with torch.no_grad():
                autoencoder_output = autoencoder(mlp_acts) 
            _, reconstructed_nll = gpt(x, y, mode="replace", replacement_tensor=autoencoder_output['reconst_acts'])
            
            # for each feature, calculate the TOTAL number of tokens on which it is active; shape: 
            feat_acts = autoencoder_output['latents'].to('cpu') # (eval_batch_size, block_size, n_features)
            torch.add(feat_acts_count, feat_acts.count_nonzero(dim=[0, 1]), out=feat_acts_count) # (n_features, )

            # calculat the AVERAGE number of non-zero entries in each feature vector and log all losses
            log_dict['losses/l0_norm'] += feat_acts.count_nonzero(dim=-1).float().mean().item()
            log_dict['losses/reconstructed_nll'] += reconstructed_nll.item()
            log_dict['losses/autoencoder_loss'] += autoencoder_output['loss'].item() 
            log_dict['losses/reconstruction_loss'] += autoencoder_output['mse_loss'].item()
            log_dict['losses/l1_norm'] += autoencoder_output['l1_loss'].item()
            log_dict['losses/nll_score'] += (nll_loss - reconstructed_nll).item()/(nll_loss - ablated_loss).item()

        # compute feature densities and plot feature density histogram
        log_feat_acts_density = np.log10(feat_acts_count[feat_acts_count != 0]/(eval_iters * eval_batch_size * gpt.config.block_size)) # (n_features,)
        feat_density_historgram = make_histogram_image(log_feat_acts_density)

        # take mean of all loss values by dividing by the number of evaluation batches; also log more metrics
        log_dict = {key: val/eval_iters for key, val in log_dict.items()}
        log_dict.update(
                {'training_step': step, 
                'training_examples': step * batch_size,
                'debug/mean_dictionary_vector_length': torch.linalg.vector_norm(autoencoder.decoder.weight, dim=0).mean(),
                'feature_density/min_log_feat_density': log_feat_acts_density.min().item() if len(log_feat_acts_density) > 0 else -100,
                'feature_density/num_neurons_with_feature_density_above_1e-3': (log_feat_acts_density > -3).sum(),
                'feature_density/num_neurons_with_feature_density_below_1e-3': (log_feat_acts_density < -3).sum(),
                'feature_density/num_neurons_with_feature_density_below_1e-4': (log_feat_acts_density < -4).sum(), 
                'feature_density/num_neurons_with_feature_density_below_1e-5': (log_feat_acts_density < -5).sum(),
                'feature_density/num_alive_neurons': len(log_feat_acts_density),
                })
        if wandb_log:
            log_dict.update({'feature_density/feature_density_histograms': wandb.Image(feat_density_historgram)})    
            wandb.log(log_dict)

        autoencoder.train()
        print(f'Exiting evaluation mode at step = {step}')
            
    ### ------------ save a checkpoint ----------- ######
    if save_checkpoint and step > 0 and (step % save_interval == 0 or step == num_steps - 1):
        checkpoint = {
                'autoencoder': autoencoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'log_dict': log_dict,
                'config': config,
                'feature_activation_counts': feat_acts_count, # may be used later to identify alive vs dead neurons
                }
        print(f"saving checkpoint to {ckpt_path} at training step = {step}")
        torch.save(checkpoint, os.path.join(ckpt_path, 'ckpt.pt'))

if wandb_log:
    wandb.finish()