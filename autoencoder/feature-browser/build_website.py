"""
Make a feature browser for a trained autoencoder model.
In this file, it is useful to keep track of shapes of each tensor. 
Each tensor is followed by a comment describing its shape.
I use the following glossary:
S: num_sampled_tokens
R: window_radius
W: window_length
L: number of autoencoder latents
H: Number of features being processed in a phase
N: total_sampled_tokens = (num_contexts * num_sampled_tokens)
T: block_size (same as nanoGPT)
B: gpt_batch_size (same as nanoGPT)
C: n_embd a.k.a d_model (same as nanoGPT)
V: vocab_size
SI: samples_per_interval

Run on a Macbook as
python build_website.py --device=cpu --dataset=shakespeare_char --gpt_ckpt_dir=out_sc_1_2_32 --sae_ckpt_dir=1712254759.95
"""

from dataclasses import dataclass
import torch
from tensordict import TensorDict 
import os
import sys 
from math import ceil
from main_page import create_main_html_page
from subpages import write_alive_feature_page, write_dead_feature_page, write_ultralow_density_feature_page

sys.path.insert(1, '../')
from resource_loader import ResourceLoader
from utils.plotting_utils import make_activations_histogram, make_logits_histogram

# hyperparameters 
# data and model
dataset = 'openwebtext' 
gpt_ckpt_dir = 'out' 
sae_ckpt_dir = 0.0 # subdirectory containing the specific model to consider
# feature page hyperparameter
num_contexts = 10000
num_sampled_tokens = 10 # number of tokens in each context on which feature activations will be computed 
window_radius = 4 # number of tokens to print on either side of a sampled token.. # V / R
num_top_activations = 10 # number of top activations for each feature 
num_intervals = 12 # number of intervals to divide activations in; = 12 in Anthropic's work
samples_per_interval = 5 # number of examples to sample from each interval of activations 
# evaluation hyperparameters
gpt_batch_size = 156 
num_phases = 52 # due to memory constraints, it's useful to process features in phases.
# system
device = 'cuda' # change it to cpu
# reproducibility
seed = 1442

@dataclass
class FeatureBrowserConfig:
    # dataset and model
    dataset: str = "openwebtext"
    gpt_ckpt_dir: str = "out" 
    sae_ckpt_dir: str = "out"
    # feature browser hyperparameters
    num_contexts: int = int(1e6)
    num_sampled_tokens: int = 10
    window_radius: int = 4
    num_top_activations: int = 10
    num_intervals: int = 12
    samples_per_interval: int = 5
    # processing hyperparameters
    seed: int = 0
    device: str = "cpu"
    gpt_batch_size: int = 156
    num_phases: int = 52
    
class FeatureBrowser(ResourceLoader):
    def __init__(self, config):
        super().__init__(
            dataset=config.dataset, 
            gpt_ckpt_dir=config.gpt_ckpt_dir,
            device=config.device,
            mode="eval",
            sae_ckpt_dir=str(config.sae_ckpt_dir),
        )
 
        # retrieve feature browser hyperparameters from config
        self.num_contexts = config.num_contexts
        self.num_sampled_tokens = config.num_sampled_tokens
        self.window_radius = config.window_radius
        self.num_top_activations = num_top_activations
        self.num_intervals = num_intervals
        self.samples_per_interval = samples_per_interval
        self.window_length = 2 * self.window_radius + 1
        self.total_sampled_tokens = self.num_contexts * self.num_sampled_tokens  # Define total sampled tokens

        self.gpt_batch_size = config.gpt_batch_size
        self.n_features = self.autoencoder.n_latents
        self.num_phases = config.num_phases 
        self.num_features_per_phase = ceil(self.n_features / self.num_phases)
        self.num_batches = ceil(self.num_contexts / self.gpt_batch_size)
        
        self.X, self.Y = self.get_text_batch(num_contexts=self.num_contexts) # sample text data for analysis # (B, T)
        self.encode, self.decode = self.load_tokenizer()
        self.html_out = os.path.join(os.path.dirname(os.path.abspath('.')), 'out', config.dataset, str(config.sae_ckpt_dir))        
        self.seed = config.seed

        # create subdirectories to store logit histograms and feature activation histograms
        os.makedirs(os.path.join(self.html_out, 'logits_histograms'), exist_ok=True)
        os.makedirs(os.path.join(self.html_out, 'activations_histograms'), exist_ok=True)

        # TODO: why are my logits of the order of 10, while Anthropic's are <1. Do they rescale them? 
        # Or is it because of linear approximation to LayerNorm?

        # self.attributed logits is a tensor of shape (n_features, vocab_size) containing logits for each feature
        self.attributed_logits = self.compute_logits()
        self.top_logits, self.bottom_logits = self.compute_top_and_bottom_logits()

        print(f"Will process features in {self.num_phases} phases. Each phase will have forward pass in {self.num_batches} batches")

        # TODO: actually, a lot more information may be initialized here.
        # e.g. context window data, perhaps current phase, which feature is being processed, top activations data for that feature, etc.
        # That way, I will not have to pass so many variables to each method.

    def build(self):
        """
        Logic: Process features in `num_phases` phases.
        In each phase, compute feature activations and feature ablations for (MLP activations of) text data `self.X`.
        Sample context windows from this data. 
        Next, use `get_top_activations` to get tokens (along with windows) with top activations for each feature.
        Note that sampled tokens are the same in all phases, thanks to the use of fn_seed in `_sample_context_windows`.
        """
        self.write_main_page() 


        for phase in range(self.num_phases):
            feature_start_idx = phase * self.num_features_per_phase
            feature_end_idx = min((phase + 1) * self.num_features_per_phase, self.n_features)
            print(f'working on features # {feature_start_idx} - {feature_end_idx} in phase {phase + 1}/{self.num_phases}')
            context_window_data = self.compute_context_window_data(feature_start_idx, feature_end_idx)
            top_acts_data = self.compute_top_activations(context_window_data)
            for h in range(0, feature_end_idx-feature_start_idx):
                # make and save histogram of logits for this feature        
                feature_id = phase * self.num_features_per_phase + h
                make_logits_histogram(logits=self.attributed_logits[feature_id, :],
                                    feature_id=feature_id,
                                    dirpath=self.html_out)
                # write the page for this feature
                self.write_feature_page(phase, h, context_window_data, top_acts_data)

            # if phase == 1:
            #     print(f'stored new feature browser pages in {self.html_out}')
            #     break

    def compute_context_window_data(self, feature_start_idx, feature_end_idx):
        """Compute data of tokens and feature activations for all context windows. 
        This should probably also include feature ablations."""
        context_window_data = self._initialize_context_window_data(feature_start_idx, feature_end_idx)

        for iter in range(self.num_batches):
            if iter % 20 == 0:
                print(f"computing feature activations for batches {iter+1}-{min(iter+20, self.num_batches)}/{self.num_batches}")
            batch_start_idx = iter * self.gpt_batch_size
            batch_end_idx = (iter + 1) * self.gpt_batch_size
            x, feature_activations, logits_difference_storage = self._compute_batch_feature_activations(batch_start_idx, 
                                                                             batch_end_idx, 
                                                                             feature_start_idx, 
                                                                             feature_end_idx)
            # x: (B, T), # feature_activations: (B, T, H), # feature_ablations: (B, T, 4C, H)
            x_context_windows, feature_acts_context_windows, logits_difference_context_window = self._sample_context_windows( x, 
                                                                                            feature_activations,  
                                                                                            logits_difference_storage,
                                                                                            fn_seed=self.seed+iter)
            # context_window_tokens: (B * S, W), context_window_feature_acts: (B * S, W, H)
            idx_start = batch_start_idx * self.num_sampled_tokens
            idx_end = batch_end_idx * self.num_sampled_tokens
            context_window_data["tokens"][idx_start:idx_end] = x_context_windows
            context_window_data["feature_acts"][idx_start:idx_end] = feature_acts_context_windows
            context_window_data["logits_diff"][idx_start:idx_end] = feature_acts_context_windows

        return context_window_data

    def compute_top_activations(self, data):
        """Computes top activations of given context window data.
        `data` is a TensorDict with keys `tokens` and `feature_acts` of shapes (B*S, W) and (B * S, W, H) respectively."""

        num_features = data["feature_acts"].shape[-1] # Label this as H.

        # Find the indices of the top activations at the center of the window
        _, top_indices = torch.topk(data["feature_acts"][:, self.window_radius, :],
                                    k=self.num_top_activations, dim=0)  # (k, H)

        # Prepare the tokens corresponding to the top activations
        top_tokens = data["tokens"][top_indices].transpose(dim0=1, dim1=2) # (k, W, H)

        # Extract and stack the top feature activations for each feature across all windows
        top_feature_activations = torch.stack(
            [data["feature_acts"][top_indices[:, i], :, i] for i in range(num_features)],
            dim=-1 
        ) # (k, W, H)

        # Bundle the top tokens and feature activations into a structured data format
        top_activations_data = TensorDict({
            "tokens": top_tokens,
            "feature_acts": top_feature_activations
        }, batch_size=[self.num_top_activations, self.window_length, num_features]) # (k, W< H)

        return top_activations_data

    @torch.no_grad()
    def compute_logits(self,):
        """
        Computes logits for each feature through path expansion approach.
        Returns a torch tensor of shape (num_features, vocab_size)
        By default, it uses full LayerNorm instead of its linear approximation. # TODO: understand if that's okay
        # also, this function is specific to SAEs trained on the activations of last MLP layer for now. TODO: generalize this
        By default, logits for each feature are shifted so that the median value is 0.
        """
        mlp_out = self.transformer.transformer.h[-1].mlp.c_proj(self.autoencoder.decoder.weight.detach().t()) # (L, C)
        ln_out = self.transformer.transformer.ln_f(mlp_out) # (L, C)
        logits = self.transformer.lm_head(ln_out) # (L, V)
        attributed_logits = (logits - logits.median(dim=1, keepdim=True).values) # (L, V)
        return attributed_logits
    
    @torch.no_grad()
    def compute_top_and_bottom_logits(self,):
        """
        Computes top and bottom logits for each feature. 
        Returns (top_logits, bottom_logits). Each is of type `torch.return_types.topk`.
        """
        # GPT-2 tokenizer has vocab size 50257. nanoGPT sets vocab size = 50304 for higher training speed.
        # See https://twitter.com/karpathy/status/1621578354024677377?lang=en
        # Decoder will give an error if a token with id > 50256 is given, and bottom_logits may pick one of these tokens. 
        # Therefore, set max token id to 50256 by hand. 
        attributed_logits = self.attributed_logits[:, :50257]
        top_logits = torch.topk(attributed_logits, largest=True, sorted=True, k=self.num_top_activations, dim=1) # (L, k)
        bottom_logits = torch.topk(attributed_logits, largest=False, sorted=True, k=self.num_top_activations, dim=1) # (L, k)
        return top_logits, bottom_logits 

    def write_feature_page(self, phase, h, data, top_acts_data):
        """"Writes features pages for dead / alive neurons; also makes a histogram.
        For alive features, it calls sample_and_write."""

        curr_feature_acts_MW = data["feature_acts"][:, :, h]
        mid_token_feature_acts_M = curr_feature_acts_MW[:, self.window_radius]
        num_nonzero_acts = torch.count_nonzero(mid_token_feature_acts_M)

        feature_id = phase * self.num_features_per_phase + h
        if num_nonzero_acts == 0:
            write_dead_feature_page(feature_id=feature_id, dirpath=self.html_out)
            return
        
        act_density = torch.count_nonzero(curr_feature_acts_MW) / (self.total_sampled_tokens * self.window_length) * 100
        non_zero_acts = curr_feature_acts_MW[curr_feature_acts_MW != 0]
        make_activations_histogram(activations=non_zero_acts, 
                       density=act_density, 
                       feature_id=feature_id,
                       dirpath=self.html_out)


        # TODO: for ultralow density and other alive neurons, I will need to give feature ablation data
        if num_nonzero_acts < self.num_intervals * self.samples_per_interval:
            write_ultralow_density_feature_page(feature_id=feature_id, 
                                                decode=self.decode,
                                                top_acts_data=top_acts_data[:num_nonzero_acts, :, h],
                                                dirpath=self.html_out)
            return

        self.sample_and_write(data, feature_id, num_nonzero_acts, mid_token_feature_acts_M, curr_feature_acts_MW, top_acts_data, h)

    def sample_and_write(self, data, feature_id, num_nonzero_acts, mid_token_feature_acts_M, curr_feature_acts_MW, top_acts_data, h):
        _, sorted_indices = torch.sort(mid_token_feature_acts_M, descending=True) # (N*S,)
        sampled_indices = torch.stack([
            j * num_nonzero_acts // self.num_intervals + 
            torch.randperm(num_nonzero_acts // self.num_intervals)[:self.samples_per_interval].sort()[0] 
            for j in range(self.num_intervals)
        ], dim=0)
        original_indices = sorted_indices[sampled_indices] # TODO: explain sampled_indices and original_indices
        sampled_acts_data = TensorDict({
            "tokens": data["tokens"][original_indices], 
            "feature_acts": curr_feature_acts_MW[original_indices],
        }, batch_size=[self.num_intervals, self.samples_per_interval, self.window_length]) # (I, SI, W)

        write_alive_feature_page(feature_id=feature_id, 
                                 decode=self.decode,
                                 top_logits=self.top_logits,
                                 bottom_logits=self.bottom_logits,
                                 top_acts_data=top_acts_data[:, :, h],
                                 sampled_acts_data=sampled_acts_data,
                                 dirpath=self.html_out)

    def _sample_context_windows(self, *args, fn_seed=0):
        """
        Select windows of tokens around randomly sampled tokens from input tensors.

        Given tensors each of shape (B, T, ...), this function returns tensors containing
        windows around randomly selected tokens. The shape of the output is (B * S, W, ...),
        where S is the number of tokens in each context to evaluate, and W is the window size
        (including the token itself and tokens on either side). By default, S = self.num_sampled_tokens,
        W = self.window_length.

        Parameters: 
        - args: Variable number of tensor arguments, each of shape (B, T, ...)
        - fn_seed (int, optional): Seed for random number generator, default is 0
        """
        if not args or not all(isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 for tensor in args):
            raise ValueError("All inputs must be torch tensors with at least 2 dimensions.")

        # Ensure all tensors have the same shape in the first two dimensions
        B, T = args[0].shape[:2]
        if not all(tensor.shape[:2] == (B, T) for tensor in args):
            raise ValueError("All tensors must have the same shape along the first two dimensions.")

        torch.manual_seed(fn_seed)
        num_sampled_tokens=self.num_sampled_tokens
        token_idx = torch.stack([self.window_radius + torch.randperm(T - 2 * self.window_radius)[:num_sampled_tokens] 
                                for _ in range(B)], dim=0) # (B, S) # use of torch.randperm for sampling without replacement
        window_idx = token_idx.unsqueeze(-1) + torch.arange(-self.window_radius, self.window_radius + 1) # (B, S, W)
        batch_idx = torch.arange(B).view(-1, 1, 1).expand_as(window_idx) # (B, S, W)

        result_tensors = []
        for tensor in args:
            if tensor.ndim == 3:
                L = tensor.shape[2]
                sliced_tensor = tensor[batch_idx, window_idx, :] # (B, S, W, L)
                sliced_tensor = sliced_tensor.view(-1, self.window_length, L) # (B *S , W, L)
            elif tensor.ndim == 2:
                sliced_tensor = tensor[batch_idx, window_idx]  # (B, S, W)
                sliced_tensor = sliced_tensor.view(-1, self.window_length) # (B*S, W)
            else:
                raise ValueError("Tensor dimensions not supported. Only 2D and 3D tensors are allowed.")
            result_tensors.append(sliced_tensor)

        return result_tensors

    def _initialize_context_window_data(self, feature_start_idx, feature_end_idx):
        num_features_in_phase = feature_end_idx - feature_start_idx
        context_window_data = TensorDict({
            "tokens": torch.zeros(self.total_sampled_tokens, self.window_length, dtype=torch.int32),
            "feature_acts": torch.zeros(self.total_sampled_tokens, self.window_length, num_features_in_phase),
            "logits_diff": torch.zeros(self.total_sampled_tokens, self.window_length, num_features_in_phase),
        }, batch_size=[self.total_sampled_tokens, self.window_length]) # (N * S, W)
        return context_window_data

    @torch.no_grad()
    def _compute_batch_feature_activations(self, batch_start_idx, batch_end_idx, feature_start_idx, feature_end_idx):
        """Computes feature activations for given batch of input text.
        """
        x = self.X[batch_start_idx:batch_end_idx].to(self.device)
        y = self.Y[batch_start_idx:batch_end_idx].to(self.device)
        B, T = x.shape
        H = feature_end_idx - feature_start_idx # number of features in this phase
        original_logits, _ = self.transformer(x, y) # original_logits.shape = (B, T, V)
        mlp_acts = self.transformer.mlp_activation_hooks[0] # (B, T, 4C)
        self.transformer.clear_mlp_activation_hooks()
        feature_activations = self.autoencoder.get_feature_activations(inputs=mlp_acts, 
                                                                    start_idx=feature_start_idx, 
                                                                    end_idx=feature_end_idx) # (B, T, H)

        dictionary_vectors = self.autoencoder.decoder.weight[:, feature_start_idx:feature_end_idx] # (4C, H)
        feature_projections = feature_activations.unsqueeze(2) * dictionary_vectors # (B, T, 4C, H)
        feature_ablations =  mlp_acts.unsqueeze(-1) - feature_projections # (B, T, 4C, H)

        logits_difference_storage = torch.zeros(B, T, H, device=self.device) # (B, T, H)
        for h in range(H):
            feat_ablation_logits, _ = self.gpt(x, y, mode="replace", replacement_tensor=feature_ablations[:, :, :, h]) # (B, T, V)

            logits_difference = original_logits - feat_ablation_logits # (B, T, V)

            # restrict logits difference to those for the next token
            logits_difference_storage[:, :, h] = logits_difference[torch.arange(B)[:, None], torch.arange(T), y] # (B, T)

        return x, feature_activations, logits_difference_storage

    def write_main_page(self):
        create_main_html_page(n_features=self.n_features, dirpath=self.html_out)

if __name__ == "__main__":

    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    configurator = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configurator.py')
    exec(open(configurator).read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------
    
    torch.manual_seed(seed)
    config = FeatureBrowserConfig(**config)
    feature_browser = FeatureBrowser(config)
    # Run the processing
    feature_browser.build()


 # TODO: tooltip css function should be imported separately and written explicitly I think, for clarity
 # TODO: methods that need to be revisited: write_feature_page, sample_and_write.
 # TODO: it would be nice if the final output does not depend on num_phases. Set seed for each feature separately?
 # TODO: we don't really need autoencoder data in eval mode. Change this in resourceloader. 