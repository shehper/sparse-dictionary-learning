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
    def __init__(self, dataset, gpt_ckpt_dir, device='cpu', mode="train"):
        assert mode in ["train", "eval", "prepare"], "mode must be train, eval or prepare"

        # directories, datasets, etc
        self.dataset = dataset # openwebtext, shakespeare_char, etc
        self.gpt_ckpt_dir = gpt_ckpt_dir # subdirectory (of transformer) contraining transformer weights
        self.device = device # for models' weights
        self.mode = mode
        self.base_dir = os.path.dirname(os.path.abspath('.'))
        # transformer model weights and dataset 
        self.text_data = self.load_text_data()
        self.transformer = self.load_transformer_model()
        self.n_ffwd = self.transformer.config.n_embd * 4
        
        if mode in ["train", "eval"]:
            self.autoencoder_data_dir = os.path.join(os.path.abspath('.'), 'data', self.dataset, f"{self.n_ffwd}")
            self.autoencoder_data = self.load_first_autoencoder_data_file()
            self.autoencoder_data_info = self.init_autoencoder_data_info()
            if mode == "eval":
                self.autoencoder = self.load_autoencoder_model()
        
    def load_text_data(self):
        text_data_path = os.path.join(self.base_dir, 'transformer', 'data', self.dataset, 'train.bin')
        text_data = np.memmap(text_data_path, dtype=np.uint16, mode='r')
        return text_data

    def load_transformer_model(self):
        ## load GPT model weights --- we need it to compute reconstruction nll and nll score
        ckpt_path = os.path.join(self.base_dir, 'transformer', self.gpt_ckpt_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        transformer = HookedGPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.' 
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        transformer.load_state_dict(state_dict)
        transformer.eval()
        transformer.to(self.device)
        return transformer

    def load_first_autoencoder_data_file(self):
        file_path = os.path.join(self.autoencoder_data_dir, '0.pt')
        assert os.path.exists(file_path), f"0.pt not found in {self.autoencoder_data_dir}"
        autoencoder_data = torch.load(file_path) 
        return autoencoder_data
    
    def get_number_of_autoencoder_data_files(self):
        try:
            num_partitions = len(next(os.walk(self.autoencoder_data_dir))[2]) # number of files in autoencoder_data_dir
        except StopIteration:
            raise ValueError("autoencoder data directory seems empty")
        return num_partitions
    
    def init_autoencoder_data_info(self):
        info = {'num_partitions': self.get_number_of_autoencoder_data_files(),
                'current_partition_id': 0,
                'offset': 0,
                'examples_per_partition': self.autoencoder_data.shape[0],
                }
        info['total_examples'] = info['examples_per_partition'] * info['num_partitions']
        return info

    def load_autoencoder_model(self):
        pass

    def get_text_batch(self, num_contexts):
        block_size = self.transformer.config.block_size
        ix = torch.randint(len(self.text_data) - block_size, (num_contexts,))
        X = torch.stack([torch.from_numpy((self.text_data[i:i+block_size]).astype(np.int64)) for i in ix])
        Y = torch.stack([torch.from_numpy((self.text_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        return X.to(device=self.device), Y.to(device=self.device)
    
    def get_autoencoder_data_batch(self, step, batch_size=8192):
        """
        Gets a batch of autoencoder data based on the current step and batch size.
        Handles the case where the new batch extends beyond the end of the current partition,
        loading the next partition if necessary.
        """

        num_partitions = self.autoencoder_data_info["num_partitions"]
        current_partition_id = self.autoencoder_data_info["current_partition_id"]
        offset = self.autoencoder_data_info["offset"]
        examples_per_partition = self.autoencoder_data_info["examples_per_partition"]

        # indices of the start and end of the batch in the current partition
        batch_start = step * batch_size - current_partition_id * examples_per_partition - offset 
        batch_end = (step + 1) * batch_size - current_partition_id * examples_per_partition - offset 
        
        # load the next partition if the end of the batch is beyond the current partition
        if batch_end > examples_per_partition and current_partition_id < num_partitions - 1:
            remaining = examples_per_partition - batch_start
            batch = self.autoencoder_data[batch_start:]
            current_partition_id += 1
            self.autoencoder_data = torch.load(f'{self.autoencoder_data_dir}/{current_partition_id}.pt')
            batch = torch.cat([batch, self.autoencoder_data[:batch_size - remaining]])
            offset = batch_size - remaining
            self.autoencoder_data_info["offset"] = offset 
            self.autoencoder_data_info["current_partition_id"] = current_partition_id 
        else:
            batch = self.autoencoder_data[batch_start:batch_end].to(torch.float32)

        assert len(batch) == batch_size, f"length of batch = {len(batch)} at step = {step} is incorrect"

        return batch.to(self.device)

    def select_resampling_data(self, size=819200):
        """
        Selects a subset of data for resampling neurons.
        """
        num_partitions = self.autoencoder_data_info["num_partitions"]
        examples_per_partition = self.autoencoder_data_info["examples_per_partition"]
        num_samples_per_partition = size // num_partitions
        resampling_data = torch.zeros(size, self.n_ffwd)
        
        for partition_id in range(num_partitions):
            partition_path = os.path.join(self.autoencoder_data_dir, f'{partition_id}.pt')
            partition_data = torch.load(partition_path)
            sample_indices = torch.randint(examples_per_partition, (num_samples_per_partition,))
            start_index = partition_id * num_samples_per_partition
            end_index = (partition_id + 1) * num_samples_per_partition
            resampling_data[start_index:end_index] = partition_data[sample_indices]

        return resampling_data