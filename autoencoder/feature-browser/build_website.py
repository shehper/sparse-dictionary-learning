"""
Make a feature browser for a trained autoencoder model.
In this file, it is useful to keep track of shapes of each tensor. 
Each tensor is followed by a comment describing its shape.
I use the following glossary:
S: num_sampled_tokens
R: window_radius
W: window_length
L: number of autoencoder latents
N: total_sampled_tokens = (num_contexts * num_sampled_tokens)
T: block_size (same as nanoGPT)
B: gpt_batch_size (same as nanoGPT)
SI: samples_per_interval

Run on a Macbook as
python build_website.py --device=cpu --dataset=shakespeare_char --gpt_ckpt_dir=out_sc_1_2_32 --autoencoder_subdir=1712254759.95
"""

import torch
from tensordict import TensorDict 
import os
import sys 
from main_page import create_main_html_page
from subpages import write_alive_feature_page, write_dead_feature_page, write_ultralow_density_feature_page

## Add path to the transformer subdirectory as it contains GPT class in model.py
sys.path.insert(0, '../../transformer')
from model import GPTConfig
from hooked_model import HookedGPT

sys.path.insert(1, '../')
from resource_loader import ResourceLoader
from utils.plotting_utils import make_histogram

# hyperparameters 
# data and model
dataset = 'openwebtext' 
gpt_ckpt_dir = 'out' 
autoencoder_subdir = 0.0 # subdirectory containing the specific model to consider
# evaluation hyperparameters
num_contexts = 10000 
gpt_batch_size = 156 # batch size for computing reconstruction nll 
# feature page hyperparameter
num_sampled_tokens = 10 # number of tokens in each context on which feature activations will be computed 
window_radius = 4 # number of tokens to print on either side of a sampled token.. # V / R
num_top_activations = 10 # number of top activations for each feature 
num_intervals = 12 # number of intervals to divide activations in; = 12 in Anthropic's work
samples_per_interval = 5 # number of examples to sample from each interval of activations 
n_features_per_phase = 20 # due to memory constraints, it's useful to process features in phases.
# system
device = 'cuda' # change it to cpu
# reproducibility
seed = 1442

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('../configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# TODO: should configurator be moved to after __name__ == __main__? Will it be executed if 
# the class is imported in another file?
# Also, we don't need config here as we are not logging anything. 
class FeatureBrowser(ResourceLoader):
    def __init__(self, dataset, gpt_ckpt_dir, device, autoencoder_subdir, num_contexts, num_sampled_tokens, 
                 window_radius, eval_batch_size, num_top_activations, num_intervals, samples_per_interval, seed):
        super().__init__(
            dataset=dataset, 
            gpt_ckpt_dir=gpt_ckpt_dir,
            device=device,
            mode="eval",
            sae_ckpt_dir=str(autoencoder_subdir),
        )
 
        self.encode, self.decode = self.load_tokenizer()
        self.n_features, self.n_ffwd = self.autoencoder.encoder.weight.shape
        self.html_out = os.path.join(os.path.dirname(os.path.abspath('.')), 'out', self.dataset, str(autoencoder_subdir))        
        self.num_contexts = num_contexts
        self.num_sampled_tokens = num_sampled_tokens
        self.window_radius = window_radius
        self.eval_batch_size = eval_batch_size
        self.num_top_activations = num_top_activations
        self.num_intervals = num_intervals
        self.samples_per_interval = samples_per_interval
        self.seed = seed

        self.total_sampled_tokens = self.num_contexts * self.num_sampled_tokens  # Define total sampled tokens

        self.n_features_per_phase = 20  # Adjust based on memory constraints
        self.n_phases = self.n_features // self.n_features_per_phase + (self.n_features % self.n_features_per_phase != 0)
        self.n_batches = self.num_contexts // self.eval_batch_size + (self.num_contexts % self.eval_batch_size != 0)
        print(f"Will process features in {self.n_phases} phases. Each phase will have forward pass in {self.n_batches} batches")

    def build(self):
        create_main_html_page(n_features=self.n_features, dirpath=self.html_out)
        X, _ = self.get_text_batch(num_contexts=self.num_contexts)
        
        for phase in range(self.n_phases):
            H = self.calculate_H(phase) # H needs to get a better name
            print(f'working on phase # {phase + 1}/{self.n_phases}: \
                  features # {phase * self.n_features_per_phase} through {phase * self.n_features_per_phase + H}')
            data = self.compute_feature_activations(phase, H, X)
            self.process_phase(data, H, phase)

            if phase == 1:
                print(f'stored new feature browser pages in {self.html_out}')
                break

    def calculate_H(self, phase):
        if phase < self.n_phases - 1:
            return self.n_features_per_phase
        else:
            return self.n_features - (phase * self.n_features_per_phase)

    def compute_feature_activations(self, phase, H, X):
        data = TensorDict({
            "tokens": torch.zeros(self.total_sampled_tokens, 2 * self.window_radius + 1, dtype=torch.int32),
            "feature_acts": torch.zeros(self.total_sampled_tokens, 2 * self.window_radius + 1, H),
        }, batch_size=[self.total_sampled_tokens, 2 * self.window_radius + 1])

        for iter in range(self.n_batches):
            print(f"Computing feature activations for batch # {iter+1}/{self.n_batches} in phase # {phase + 1}")
            start_idx = iter * self.eval_batch_size
            end_idx = (iter + 1) * self.eval_batch_size
            x = X[start_idx:end_idx].to(self.device)
            _, _ = self.transformer(x)
            mlp_acts_BTF = self.transformer.mlp_activation_hooks[0]
            self.transformer.clear_mlp_activation_hooks()
            feature_acts_BTH = self.autoencoder.get_feature_activations(inputs=mlp_acts_BTF, 
                                                                        start_idx=phase*H, 
                                                                        end_idx=(phase+1)*H)
            X_PW, feature_acts_PWH = self.select_context_windows(x, feature_acts_BTH, 
                                                            num_sampled_tokens=self.num_sampled_tokens, 
                                                            window_radius=self.window_radius, 
                                                            fn_seed=self.seed+iter)
            idx_start = iter * self.eval_batch_size * self.num_sampled_tokens
            idx_end = (iter + 1) * self.eval_batch_size * self.num_sampled_tokens
            data["tokens"][idx_start:idx_end] = X_PW
            data["feature_acts"][idx_start:idx_end] = feature_acts_PWH

        return data

    def process_phase(self, data, H, phase):
        print(f'computing top k feature activations in phase # {phase + 1}/{self.n_phases}')
        _, topk_indices_kH = torch.topk(data["feature_acts"][:, self.window_radius, :], k=self.num_top_activations, dim=0)
        top_acts_data_kWH = TensorDict({
            "tokens": data["tokens"][topk_indices_kH].transpose(dim0=1, dim1=2),
            "feature_acts": torch.stack([data["feature_acts"][topk_indices_kH[:, i], :, i] for i in range(H)], dim=-1)
        }, batch_size=[self.num_top_activations, 2 * self.window_radius + 1, H])

        for h in range(H):
            self.process_feature(data, H, phase, h, top_acts_data_kWH)

    def process_feature(self, data, H, phase, h, top_acts_data_kWH):
        curr_feature_acts_MW = data["feature_acts"][:, :, h]
        mid_token_feature_acts_M = curr_feature_acts_MW[:, self.window_radius]
        num_nonzero_acts = torch.count_nonzero(mid_token_feature_acts_M)

        feature_id = phase * self.n_features_per_phase + h
        if num_nonzero_acts == 0:
            write_dead_feature_page(feature_id=feature_id, dirpath=self.html_out)
            return
        
        act_density = torch.count_nonzero(curr_feature_acts_MW) / (self.total_sampled_tokens * (2 * self.window_radius + 1)) * 100
        non_zero_acts = curr_feature_acts_MW[curr_feature_acts_MW != 0]
        make_histogram(activations=non_zero_acts, 
                       density=act_density, 
                       feature_id=feature_id,
                       dirpath=self.html_out)

        if num_nonzero_acts < self.num_intervals * self.samples_per_interval:
            write_ultralow_density_feature_page(feature_id=feature_id, 
                                                decode=self.decode,
                                                top_acts_data=top_acts_data_kWH[:num_nonzero_acts, :, h],
                                                dirpath=self.html_out)
            return

        self.sample_and_write(data, feature_id, num_nonzero_acts, mid_token_feature_acts_M, curr_feature_acts_MW, top_acts_data_kWH, h)

    def sample_and_write(self, data, feature_id, num_nonzero_acts, mid_token_feature_acts_M, curr_feature_acts_MW, top_acts_data_kWH, h):
        sorted_acts_M, sorted_indices_M = torch.sort(mid_token_feature_acts_M, descending=True)
        sampled_indices = torch.stack([
            j * num_nonzero_acts // self.num_intervals + 
            torch.randperm(num_nonzero_acts // self.num_intervals)[:self.samples_per_interval].sort()[0] 
            for j in range(self.num_intervals)
        ], dim=0)
        original_indices = sorted_indices_M[sampled_indices]
        sampled_acts_data_IXW = TensorDict({
            "tokens": data["tokens"][original_indices],
            "feature_acts": curr_feature_acts_MW[original_indices],
        }, batch_size=[self.num_intervals, self.samples_per_interval, 2 * self.window_radius + 1])

        write_alive_feature_page(feature_id=feature_id, 
                                 decode=self.decode,
                                 top_acts_data=top_acts_data_kWH[:, :, h],
                                 sampled_acts_data=sampled_acts_data_IXW,
                                 dirpath=self.html_out)

    @staticmethod
    def select_context_windows(*args, num_sampled_tokens, window_radius, fn_seed=0):
        """
        Select windows of tokens around randomly sampled tokens from input tensors.

        Given tensors each of shape (B, T, ...), this function returns tensors containing
        windows around randomly selected tokens. The shape of the output is (B * S, W, ...),
        where S is the number of tokens in each context to evaluate, and W is the window size
        (including the token itself and tokens on either side).

        Parameters:
        - args: Variable number of tensor arguments, each of shape (B, T, ...)
        - num_sampled_tokens (int): The number of tokens in each context on which to evaluate
        - window_radius (int): The number of tokens on either side of the sampled token
        - fn_seed (int, optional): Seed for random number generator, default is 0

        Returns:
        - A list of tensors, each of shape (B * S, W, ...), where S is `num_sampled_tokens` and W is
        the window size calculated as 2 * `window_radius` + 1.

        Raises:
        - AssertionError: If no tensors are provided, or if the tensors do not have the required shape.

        Example usage:
        ```
        tensor1 = torch.randn(10, 20, 30)  # Example tensor
        windows = select_context_windows(tensor1, num_sampled_tokens=5, window_radius=2)
        ```
        """
        if not args or not all(isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 for tensor in args):
            raise ValueError("All inputs must be torch tensors with at least 2 dimensions.")

        # Ensure all tensors have the same shape in the first two dimensions
        B, T = args[0].shape[:2]
        if not all(tensor.shape[:2] == (B, T) for tensor in args):
            raise ValueError("All tensors must have the same shape along the first two dimensions.")

        torch.manual_seed(fn_seed)
        window_length = 2 * window_radius + 1
        token_idx = torch.stack([window_radius + torch.randperm(T - 2 * window_radius)[:num_sampled_tokens] 
                                for _ in range(B)], dim=0) # (B, S)
        window_idx = token_idx.unsqueeze(-1) + torch.arange(-window_radius, window_radius + 1) # (B, S, W)
        batch_idx = torch.arange(B).view(-1, 1, 1).expand_as(window_idx) # (B, S, W)

        result_tensors = []
        for tensor in args:
            if tensor.ndim == 3:
                L = tensor.shape[2]
                sliced_tensor = tensor[batch_idx, window_idx, :] # (B, S, W, L)
                sliced_tensor = sliced_tensor.view(-1, window_length, L) # (B *S , W, L)
            elif tensor.ndim == 2:
                sliced_tensor = tensor[batch_idx, window_idx]  # (B, S, W)
                sliced_tensor = sliced_tensor.view(-1, window_length) # (B*S, W)
            else:
                raise ValueError("Tensor dimensions not supported. Only 2D and 3D tensors are allowed.")
            result_tensors.append(sliced_tensor)

        return result_tensors

if __name__ == "__main__":

    torch.manual_seed(seed)
    feature_browser = FeatureBrowser(
        dataset=dataset, 
        gpt_ckpt_dir=gpt_ckpt_dir,
        device=device,
        autoencoder_subdir=autoencoder_subdir,
        num_contexts=num_contexts,
        num_sampled_tokens=num_sampled_tokens,
        window_radius=window_radius,
        eval_batch_size=gpt_batch_size,
        num_top_activations=num_top_activations,
        num_intervals=num_intervals,
        samples_per_interval=samples_per_interval,
        seed=seed
    )

    # Run the processing
    feature_browser.build()






