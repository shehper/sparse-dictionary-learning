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
        self.curr_dir = os.path.abspath('.') # after converting code to a package, this might not be needed
        self.gpt_dir = gpt_dir # subdirectory (of transformer) contraining transformer weights
        self.device = device # for models' weights; (large) data is mostly stored in CPU RAM

        # models (to be loaded from checkpoints)
        self.transformer = None
        self.autoencoder = None 

        # datasets 
        self.text_data = None
        self.autoencoder_data = None
        self.resampling_data = None
        self.eval_data = None # a piece of 
        # self.resample_data_size = resample_data_size # as by Anthropic

        # autoencoder dataset might be stored in more than one files
        # in this case use
        self.n_partitions = 0
        self.curr_partition_id = 0
        self.num_examples_per_partition = 0 # assuming all files have the same number of examples
        self.num_examples_total = 0
        self.offset = 0 # sometimes, when we load a new file, we will use `a` examples from one (previous) partition
        # and `b` examples from another (the new) partition. offset is the number `b`.
        self.batch_size = batch_size 
        
    def load_text_data(self):
        ## load tokenized text data
        self.text_data_dir = os.path.join(os.path.dirname(self.curr_dir), 'transformer', 'data', self.dataset)
        self.text_data = np.memmap(os.path.join(self.text_data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        return self.text_data

    def load_transformer_model(self):
        ## load GPT model weights --- we need it to compute reconstruction nll and nll score
        ckpt_path = os.path.join(os.path.dirname(self.curr_dir), 'transformer', self.gpt_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        gpt = HookedGPT(gptconf)
        state_dict = checkpoint['model']
        compile = False # TODO: Don't know why I needed to set compile to False before loading the model..
        # TODO: Also, I dont know why the next 4 lines are needed. state_dict does not seem to have any keys with unwanted_prefix.
        unwanted_prefix = '_orig_mod.' 
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        gpt.load_state_dict(state_dict)
        gpt.eval()
        gpt.to(self.device)
        if compile:
            gpt = torch.compile(gpt) # requires PyTorch 2.0 (optional)
        self.transformer = gpt
        self.block_size = gpt.config.block_size
        self.n_ffwd = 4 * gpt.config.n_embd
        return self.transformer 

    def load_autoencoder_data(self):
        self.autoencoder_data_dir = os.path.join(os.path.abspath('.'), 'data', self.dataset, f"{self.n_ffwd}")
        self.n_partitions = len(next(os.walk(self.autoencoder_data_dir))[2]) # number of files in autoencoder_data_dir
        self.autoencoder_data = torch.load(f'{self.autoencoder_data_dir}/{self.curr_partition_id}.pt') # current partition
        if self.curr_partition_id == 0: # only need to compute these variables once
            self.num_examples_per_partition = self.autoencoder_data.shape[0]
            self.num_examples_total = self.num_examples_per_partition * self.n_partitions
        return self.autoencoder_data
    
    def load_resampling_data(self):
        self.resampling_data = torch.load(f'{self.autoencoder_data_dir}/resampling_data/data.pt')# TODO: make sure this name is changed to sae_data/resampling_data.pt
        return self.resampling_data

    def get_text_batch(self, num_contexts):
        ix = torch.randint(len(self.text_data) - self.block_size, (num_contexts,))
        X = torch.stack([torch.from_numpy((self.text_data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        Y = torch.stack([torch.from_numpy((self.text_data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
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

    # def select_resampling_data(self):
    #     resample_data = torch.zeros(self.resample_data_size, dtype=torch.float32)
    #     examples_per_partition = self.resample_data_size // self.n_partitions
    #     for partition_index in range(self.n_partitions):
    #         partition = torch.load(os.path.join(data_dir, f'{partition_index}.pt')) # current partition
    #         print(f'working on partition # {partition_index}')
    #         partition_size = partition.shape[0]
    #         ix = torch.randint(partition_size, (examples_per_partition,)) # pick examples_per_partition examples from current partition
    #         out = torch.cat([out, partition[ix]]) # include them in output tensor 
    #         print(f'Length of data after working on partition # {partition_index} = {out.shape[0]}')