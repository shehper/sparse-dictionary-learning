import torch
import os
data_dir = 'sae_data'
total_partitions = len(next(os.walk(data_dir))[2]) # number of partitions of (or files in) sae_data
out_file = 'sae_data_for_resampling_neurons.pt'

# set aside memory
examples_needed = 4*819200 
examples_needed_per_partition = examples_needed // total_partitions
n_ffwd = 512
out = torch.tensor([], dtype=torch.float16)

for partition_index in range(total_partitions):
    partition = torch.load(f'sae_data/sae_data_{partition_index}.pt') # current partition
    print(f'working on partition # {partition_index}')
    examples_per_partition = partition.shape[0]
    ix = torch.randint(examples_per_partition, (examples_needed_per_partition,))
    out = torch.cat([out, partition[ix]])
    print(f'Length of data after working on partition # {partition_index} = {out.shape[0]}')

torch.save(out, out_file)
print(f'saved data in {out_file}; shape of data: {out.shape}')
