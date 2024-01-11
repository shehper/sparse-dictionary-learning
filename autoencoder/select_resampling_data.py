import torch
import os
data_dir = 'sae_data'
out_file = 'data_for_resampling_neurons.pt'
num_examples = 4*819200 
n_ffwd = 512

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

total_partitions = len(next(os.walk(data_dir))[2]) # number of files in data_dir (= sae_data by default)
examples_per_partition = num_examples // total_partitions

# initiate output tensor
out = torch.tensor([], dtype=torch.float16)

for partition_index in range(total_partitions):
    partition = torch.load(f'sae_data/sae_data_{partition_index}.pt') # current partition
    print(f'working on partition # {partition_index}')
    examples_per_partition = partition.shape[0]
    ix = torch.randint(examples_per_partition, (num_examples,)) # pick examples_per_partition examples from current partition
    out = torch.cat([out, partition[ix]]) # include them in output tensor 
    print(f'Length of data after working on partition # {partition_index} = {out.shape[0]}')

torch.save(out, out_file)
print(f'saved data in {out_file}; shape of data: {out.shape}')
