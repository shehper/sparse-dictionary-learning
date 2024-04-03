"""
ResourceLoader class takes care of loading model weights, datasets and getting batches of training/eval data from datasets.
"""

import os 
import numpy as np 
import torch 
import sys

## Add path to the transformer subdirectory as it contains GPT class in model.py
sys.path.insert(0, '../transformer')
from model import GPTConfig
from hooked_model import HookedGPT

class ResourceLoader:
    def __init__(self, dataset, gpt_dir, batch_size=8192, device='cpu'):
        # directories, datasets, etc
        self.dataset = dataset # openwebtext, shakespeare_char, etc
        self.gpt_dir = gpt_dir # subdirectory (of transformer) contraining transformer weights
        self.device = device # for models' weights; (large) data is mostly stored in CPU RAM

        # models (to be loaded from checkpoints)
        self.transformer = None
        self.autoencoder = None 

        # datasets 
        self.text_data = None
        self.autoencoder_data = None
        self.resampling_data = None
        # self.resample_data_size = resample_data_size # as by Anthropic

        # autoencoder dataset might be stored in more than one files
        # in this case use
        self.n_partitions = 0
        self.curr_partition_id = 0
        self.num_examples_per_partition = 0 # assuming all files have the same number of examples
        self.num_examples_total = 0 # TODO: fix this
        self.offset = 0 # sometimes, when we load a new file, we will use `a` examples from one (previous) partition
        # and `b` examples from another (the new) partition. offset is the number `b`.
        self.batch_size = batch_size 
        
    def load_text_data(self):
        parent_dir = os.path.dirname(os.path.abspath('.'))
        text_data_path = os.path.join(parent_dir, 'transformer', 'data', self.dataset, 'train.bin')
        self.text_data = np.memmap(text_data_path, dtype=np.uint16, mode='r')
        return self.text_data

    def load_transformer_model(self):
        ## load GPT model weights --- we need it to compute reconstruction nll and nll score
        parent_dir = os.path.dirname(os.path.abspath('.'))
        ckpt_path = os.path.join(parent_dir, 'transformer', self.gpt_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        self.transformer = HookedGPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.' 
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.transformer.load_state_dict(state_dict)
        self.transformer.eval()
        self.transformer.to(self.device)
        self.n_ffwd = self.transformer.config.n_embd * 4
        return self.transformer

    def load_autoencoder_data(self):
        self.autoencoder_data_dir = os.path.join(os.path.abspath('.'), 'data', self.dataset, f"{self.n_ffwd}")
        self.n_partitions = len(next(os.walk(self.autoencoder_data_dir))[2]) # number of files in autoencoder_data_dir
        self.autoencoder_data = torch.load(f'{self.autoencoder_data_dir}/{self.curr_partition_id}.pt') # current partition
        if self.curr_partition_id == 0: # only need to compute these variables once
            self.num_examples_per_partition = self.autoencoder_data.shape[0]
            self.num_examples_total = self.num_examples_per_partition * self.n_partitions
        return self.autoencoder_data

    def get_text_batch(self, num_contexts):
        block_size = self.transformer.config.block_size
        ix = torch.randint(len(self.text_data) - block_size, (num_contexts,))
        X = torch.stack([torch.from_numpy((self.text_data[i:i+block_size]).astype(np.int64)) for i in ix])
        Y = torch.stack([torch.from_numpy((self.text_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        return X.to(device=self.device), Y.to(device=self.device)
    
    def get_autoencoder_data_batch(self, step):
        # A custom data loader specific for our needs.  
        # It assumes that data is stored in multiple files in the 'sae_data' folder. # TODO: folder name might be changed later
        # It selects a batch from 'current_partition' sequentially. When 'current_partition' reaches its end, it loads the next partition. 
        # Input: step: current training step
        # Returns: batch: batch of data
        batch_start = step * self.batch_size - self.curr_partition_id * self.num_examples_per_partition - self.offset # index of the start of the batch in the current partition
        batch_end = (step + 1) * self.batch_size - self.curr_partition_id * self.num_examples_per_partition - self.offset # index of the end of the batch in the current partition
        # check if the end of the batch is beyond the current partition
        if batch_end > self.num_examples_per_partition and self.curr_partition_id < self.n_partitions - 1:
            # handle transition to next part
            remaining = self.num_examples_per_partition - batch_start
            batch = self.autoencoder_data[batch_start:].to(torch.float32)
            self.curr_partition_id += 1
            self.autoencoder_data = torch.load(f'{self.autoencoder_data_dir}/{self.curr_partition_id}.pt')
            print(f'partition = {self.curr_partition_id} of training data successfully loaded!')
            batch = torch.cat([batch, self.autoencoder_data[:self.batch_size - remaining]]).to(torch.float32)
            self.offset = self.batch_size - remaining
        else:
            # normal batch processing
            batch = self.autoencoder_data[batch_start:batch_end].to(torch.float32)
        assert len(batch) == self.batch_size, f"length of batch = {len(batch)} at step = {step} and partition number = {self.curr_partition_id} is not correct"
        return batch.to(self.device)

    def select_resampling_data(self, size=819200):
        """
        Selects a subset of data for resampling neurons.
        """
        num_samples_per_partition = size // self.n_partitions
        resampling_data = torch.zeros(size, self.n_ffwd)
        
        for partition_id in range(self.n_partitions):
            partition_path = os.path.join(self.autoencoder_data_dir, f'{partition_id}.pt')
            partition_data = torch.load(partition_path)
            sample_indices = torch.randint(self.num_examples_per_partition, (num_samples_per_partition,))
            start_index = partition_id * num_samples_per_partition
            end_index = (partition_id + 1) * num_samples_per_partition
            resampling_data[start_index:end_index] = partition_data[sample_indices]

        self.resampling_data = resampling_data
        return resampling_data