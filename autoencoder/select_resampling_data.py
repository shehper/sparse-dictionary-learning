"""
Pre-select data for neuron resampling during autoencoder training

Run on a Macbook as
python -u select_resampling_data.py --resampling_data_size=200 --dataset=shakespeare_char 
"""

import torch
import os
# give dataset and n_ffwd to access the correct data directory from which we resample data
dataset = 'openwebtext'
n_ffwd = 512
resampling_data_size = 4*819200 # ~ 3.3M

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# directory where data is stored
data_dir = os.path.join(os.path.abspath('.'), 'data', dataset, f"{n_ffwd}")

# compute the total number of files in data_dir 
try:
    n_partitions = len(next(os.walk(data_dir))[2]) # number of files in data_dir
    print(f"found data stored in {n_partitions} files in {data_dir}")
except StopIteration:
    raise ValueError(f"""No files found in {data_dir}. 
                    Make sure to save training data in the correct directory and pass correct dataset and n_ffwd arguments.""")

# compute the number of examples to be sampled from each partition
examples_per_partition = resampling_data_size // n_partitions

# initiate output tensor
# TODO: initiate out to have the shape we want for cleaner code 
# TODO: also this code could be placed in a resource_loader class in the future. 
# TODO: and instead of running this file separately, we could just call get_resampling_data() in the beginning of train.py
out = torch.tensor([], dtype=torch.float16)
for partition_index in range(n_partitions):
    partition = torch.load(os.path.join(data_dir, f'{partition_index}.pt')) # current partition
    print(f'working on partition # {partition_index}')
    partition_size = partition.shape[0]
    ix = torch.randint(partition_size, (examples_per_partition,)) # pick examples_per_partition examples from current partition
    out = torch.cat([out, partition[ix]]) # include them in output tensor 
    print(f'Length of data after working on partition # {partition_index} = {out.shape[0]}')

os.makedirs(os.path.join(data_dir, 'resampling_data'), exist_ok=True)
out_file = os.path.join(data_dir, 'resampling_data', 'data.pt')
torch.save(out, out_file)
print(f'saved resampling data in {out_file}; shape of data: {out.shape}')
