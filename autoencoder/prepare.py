import os
import torch
import time
import psutil
from resource_loader import ResourceLoader

# Default parameters, can be overridden by command line arguments or a configuration file
seed = 0
device = 'cpu'
total_contexts = int(2e6)  # Number of context windows
tokens_per_context = 200  # Tokens per context window
dataset = 'openwebtext'
gpt_ckpt_dir = 'out'  # Model checkpoint directory
n_files = 20  # Number of output files

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

torch.manual_seed(seed)

# Load resources and model
resource_loader = ResourceLoader(dataset=dataset, gpt_ckpt_dir=gpt_ckpt_dir, mode="prepare")
gpt_model = resource_loader.transformer

# Get model configurations
block_size = gpt_model.config.block_size
n_ffwd = 4 * gpt_model.config.n_embd 

# Prepare storage for activations
data_storage = torch.zeros(total_contexts * tokens_per_context, n_ffwd, dtype=torch.float32)
shuffled_indices = torch.randperm(total_contexts * tokens_per_context)

def compute_activations():
    start_time = time.time()
    gpt_batch_size = 500
    n_batches = total_contexts // gpt_batch_size

    for batch in range(n_batches):
        # Load batch and compute activations
        x, _ = resource_loader.get_text_batch(gpt_batch_size)
        _, _ = gpt_model(x)  # Forward pass
        activations = gpt_model.mlp_activation_hooks[0]  # Retrieve activations

        # Clean up to save memory
        gpt_model.clear_mlp_activation_hooks()

        # Process and store activations
        token_locs = torch.randint(block_size, (gpt_batch_size, tokens_per_context))
        data = torch.gather(activations, 1, token_locs.unsqueeze(2).expand(-1, -1, activations.size(2))).view(-1, n_ffwd)
        data_storage[shuffled_indices[batch * gpt_batch_size * tokens_per_context: (batch + 1) * gpt_batch_size * tokens_per_context]] = data

        print(f"Batch {batch}/{n_batches} processed in {(time.time() - start_time) / (batch + 1):.2f} seconds; "
              f"Memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB available, {psutil.virtual_memory().percent}% used.")

def save_activations():
    autoencoder_data_dir = os.path.join(os.path.abspath('.'), 'data', dataset, str(n_ffwd))
    os.makedirs(autoencoder_data_dir, exist_ok=True)
    examples_per_file = total_contexts * tokens_per_context // n_files

    for i in range(n_files):
        file_path = f'{autoencoder_data_dir}/{seed * n_files + i}.pt'
        if os.path.exists(file_path):
            print(f"Warning: File {file_path} already exists and will be overwritten.")

        # Save data to file, cloning to reduce memory usage
        torch.save(data_storage[i * examples_per_file: (i + 1) * examples_per_file].clone(), file_path)
        print(f'Saved {file_path}')

if __name__ == '__main__':
    compute_activations()
    save_activations()
