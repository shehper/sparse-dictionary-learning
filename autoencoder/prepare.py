""""
Prepares training dataset for our autoencoder. 
Run on Macbook as
python -u prepare.py --num_contexts=5000 --num_sampled_tokens=16 --dataset=shakespeare_char --gpt_ckpt_dir=out_sc_1_2_32
"""
import os
import torch
import time
import psutil
from resource_loader import ResourceLoader

# Default parameters, can be overridden by command line arguments or a configuration file
# dataset and model
dataset = 'openwebtext'
gpt_ckpt_dir = 'out'  # Model checkpoint directory
# autoencoder data size
num_contexts = int(2e6)  # Number of context windows
num_sampled_tokens = 200  # Tokens per context window
# system
device = 'cpu'
num_partitions = 20  # Number of output files
# reproducibility
seed = 0

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

torch.manual_seed(seed)

# Load resources and model
resource_loader = ResourceLoader(dataset=dataset, gpt_ckpt_dir=gpt_ckpt_dir, mode="prepare")
gpt = resource_loader.transformer

# Get model configurations
block_size = gpt.config.block_size
n_ffwd = 4 * gpt.config.n_embd 

# Prepare storage for activations
data_storage = torch.zeros(num_contexts * num_sampled_tokens, n_ffwd, dtype=torch.float32)
shuffled_indices = torch.randperm(num_contexts * num_sampled_tokens)

def compute_activations():
    start_time = time.time()
    gpt_batch_size = 500
    n_batches = num_contexts // gpt_batch_size

    for batch in range(n_batches):
        # Load batch and compute activations
        x, _ = resource_loader.get_text_batch(gpt_batch_size)
        _, _ = gpt(x)  # Forward pass
        activations = gpt.mlp_activation_hooks[0]  # Retrieve activations

        # Clean up to save memory
        gpt.clear_mlp_activation_hooks()

        # Process and store activations
        token_locs = torch.stack([torch.randperm(block_size)[:num_sampled_tokens] for _ in range(gpt_batch_size)])
        data = torch.gather(activations, 1, token_locs.unsqueeze(2).expand(-1, -1, activations.size(2))).view(-1, n_ffwd)
        data_storage[shuffled_indices[batch * gpt_batch_size * num_sampled_tokens: (batch + 1) * gpt_batch_size * num_sampled_tokens]] = data

        print(f"Batch {batch}/{n_batches} processed in {(time.time() - start_time) / (batch + 1):.2f} seconds; "
              f"Memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB available, {psutil.virtual_memory().percent}% used.")

def save_activations():
    sae_data_dir = os.path.join(os.path.abspath('.'), 'data', dataset, str(n_ffwd))
    os.makedirs(sae_data_dir, exist_ok=True)
    examples_per_file = num_contexts * num_sampled_tokens // num_partitions

    for i in range(num_partitions):
        file_path = f'{sae_data_dir}/{seed * num_partitions + i}.pt'
        if os.path.exists(file_path):
            print(f"Warning: File {file_path} already exists and will be overwritten.")

        # Save data to file, cloning to reduce memory usage
        torch.save(data_storage[i * examples_per_file: (i + 1) * examples_per_file].clone(), file_path)
        print(f'Saved {file_path}')

if __name__ == '__main__':
    compute_activations()
    save_activations()
