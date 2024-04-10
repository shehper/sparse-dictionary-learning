import os
import numpy as np
import torch
import sys
import pickle
import tiktoken

# Extend the Python path to include the transformer subdirectory for GPT class import
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'transformer'))
from model import GPTConfig
from hooked_model import HookedGPT

class ResourceLoader:
    """
    Manages resources for training, evaluation, or preparation.
    This includes loading datasets, model weights, and handling batches of data.
    """

    def __init__(self, dataset, gpt_ckpt_dir, device='cpu', mode="train", sae_ckpt_dir=""):
        assert mode in ["train", "eval", "prepare"], "Invalid mode; must be 'train', 'eval', or 'prepare'."

        self.dataset = dataset  # Name of the dataset (e.g., openwebtext, shakespeare_char)
        self.gpt_ckpt_dir = gpt_ckpt_dir  # Directory containing GPT model weights
        self.device = device  # Device on which the models will be loaded
        self.mode = mode
        
        # Set the path to the repository as the base directory
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(current_file_dir)

        # Load the text data and transformer model
        self.text_data = self.load_text_data()
        self.transformer = self.load_transformer_model()
        self.n_ffwd = self.transformer.config.n_embd * 4

        if mode == "train":
            self.autoencoder_data_dir = os.path.join(self.base_dir, 'autoencoder', 'data', self.dataset, str(self.n_ffwd))
            self.autoencoder_data = self.load_next_autoencoder_partition(partition_id=0)
            self.autoencoder_data_info = self.init_autoencoder_data_info()
            
        if mode == "eval":
            assert sae_ckpt_dir, "A path to autoencoder checkpoint must be given"
            self.sae_ckpt_dir = sae_ckpt_dir
            self.autoencoder = self.load_autoencoder_model()
        
    def load_text_data(self):
        """Loads the text data from the specified dataset."""
        text_data_path = os.path.join(self.base_dir, 'transformer', 'data', self.dataset, 'train.bin')
        return np.memmap(text_data_path, dtype=np.uint16, mode='r')

    def load_transformer_model(self):
        """Loads the GPT model with pre-trained weights."""
        ckpt_path = os.path.join(self.base_dir, 'transformer', self.gpt_ckpt_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        gpt_conf = GPTConfig(**checkpoint['model_args'])
        transformer = HookedGPT(gpt_conf)
        state_dict = checkpoint['model']

        # Remove unwanted prefix from state_dict keys
        unwanted_prefix = '_orig_mod.' 
        for key in list(state_dict.keys()):
            if key.startswith(unwanted_prefix):
                state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

        transformer.load_state_dict(state_dict)
        transformer.eval()
        transformer.to(self.device)
        return transformer
    
    def get_number_of_autoencoder_data_files(self):
        """Returns the number of files in the autoencoder data directory."""
        try:
            num_partitions = len(next(os.walk(self.autoencoder_data_dir))[2])
        except StopIteration:
            raise ValueError("Autoencoder data directory is empty")
        return num_partitions
    
    def init_autoencoder_data_info(self):
        """Initializes and returns information about the autoencoder data."""
        num_partitions = self.get_number_of_autoencoder_data_files()
        return {
            'num_partitions': num_partitions,
            'current_partition_id': 0,
            'offset': 0,
            'examples_per_partition': self.autoencoder_data.shape[0],
            'total_examples': num_partitions * self.autoencoder_data.shape[0]
        }

    def load_autoencoder_model(self):
        """Loads the AutoEncoder model with pre-trained weights"""
        autoencoder_path = os.path.join(self.base_dir, "autoencoder", "out", self.dataset, self.sae_ckpt_dir)
        autoencoder_ckpt = torch.load(os.path.join(autoencoder_path, 'ckpt.pt'), map_location=self.device)
        state_dict = autoencoder_ckpt['autoencoder']
        n_features, n_ffwd = state_dict['encoder.weight'].shape # H, F
        l1_coeff = autoencoder_ckpt['config']['l1_coeff']
        from autoencoder import AutoEncoder
        autoencoder = AutoEncoder(n_ffwd, n_features, lam=l1_coeff).to(self.device)
        autoencoder.load_state_dict(state_dict)
        return autoencoder

    def get_text_batch(self, num_contexts):
        """Generates and returns a batch of text data for training or evaluation."""
        block_size = self.transformer.config.block_size
        ix = torch.randint(len(self.text_data) - block_size, (num_contexts,))
        X = torch.stack([torch.from_numpy(self.text_data[i:i+block_size].astype(np.int64)) for i in ix])
        Y = torch.stack([torch.from_numpy(self.text_data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        return X.to(device=self.device), Y.to(device=self.device)
    
    def get_autoencoder_data_batch(self, step, batch_size=8192):
        """
        Retrieves a batch of autoencoder data based on the step and batch size.
        It loads the next data partition if the batch exceeds the current partition.
        """
        info = self.autoencoder_data_info
        batch_start = step * batch_size - info["current_partition_id"] * info["examples_per_partition"] - info["offset"]
        batch_end = batch_start + batch_size

        if batch_end > info["examples_per_partition"]:
            # When batch exceeds current partition, load data from the next partition
            if info["current_partition_id"] < info["num_partitions"] - 1:
                remaining = info["examples_per_partition"] - batch_start
                batch = self.autoencoder_data[batch_start:]
                info["current_partition_id"] += 1
                self.load_next_autoencoder_partition(info["current_partition_id"])
                batch = torch.cat([batch, self.autoencoder_data[:batch_size - remaining]])
                info["offset"] = batch_size - remaining
            else:
                raise IndexError("Autoencoder data batch request exceeds available partitions.")
        else:
            batch = self.autoencoder_data[batch_start:batch_end]

        assert len(batch) == batch_size, f"Batch length mismatch at step {step}"
        return batch.to(self.device)

    def load_next_autoencoder_partition(self, partition_id):
        """
        Loads the specified partition of the autoencoder data.
        """
        file_path = os.path.join(self.autoencoder_data_dir, f'{partition_id}.pt')
        self.autoencoder_data = torch.load(file_path)
        return self.autoencoder_data

    def select_resampling_data(self, size=819200):
        """
        Selects a subset of autoencoder data for resampling, distributed evenly across partitions.
        """
        info = self.autoencoder_data_info
        num_samples_per_partition = size // info["num_partitions"]
        resampling_data = torch.zeros(size, self.n_ffwd)
        
        for partition_id in range(info["num_partitions"]):
            partition_data = torch.load(os.path.join(self.autoencoder_data_dir, f'{partition_id}.pt'))
            sample_indices = torch.randint(info["examples_per_partition"], (num_samples_per_partition,))
            start_index = partition_id * num_samples_per_partition
            resampling_data[start_index:start_index + num_samples_per_partition] = partition_data[sample_indices]

        return resampling_data

    def load_tokenizer(self):
        load_meta = False
        meta_path = os.path.join(self.base_dir, 'transformer', 'data', self.dataset, 'meta.pkl')
        load_meta = os.path.exists(meta_path)
        if load_meta:
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            # TODO want to make this more general to arbitrary encoder/decoder schemes
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
        else:
            # ok let's assume gpt-2 encodings by default
            print("No meta.pkl found, assuming GPT-2 encodings...")
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)
        return encode, decode