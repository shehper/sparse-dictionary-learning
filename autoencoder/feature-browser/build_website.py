"""
Make a feature browser for a trained autoencoder model.
In this file, it is useful to keep track of shapes of each tensor. 
Each tensor is followed by a comment describing its shape.
I use the following glossary:
S: num_sampled_tokens
R: window_radius
W: window_length
L: number of autoencoder latents
N: num_contexts
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
import gc
from helper_functions import select_context_windows
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
dataset = 'openwebtext' # TODO: dataset should probably be called gpt_dataset.
gpt_ckpt_dir = 'out' # TODO: autoencoder_subdir should be renamed sae_ckpt_dir. It should be a subdirectory of autoencoder/out/gpt_dataset
autoencoder_subdir = 0.0 # subdirectory containing the specific model to consider # TODO: might have to think about it. It shouldn't be a float.
# evaluation hyperparameters
num_contexts = 10000 
eval_batch_size = 156 # batch size for computing reconstruction nll # TODO: call it gpt_batch_size 
# feature page hyperparameter
num_sampled_tokens = 10 # number of tokens in each context on which feature activations will be computed # S
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

torch.manual_seed(seed)
resourceloader = ResourceLoader(
                        dataset=dataset, 
                        gpt_ckpt_dir=gpt_ckpt_dir,
                        device=device,
                        mode="eval",
                        sae_ckpt_dir=str(autoencoder_subdir),
                        )

autoencoder = resourceloader.autoencoder
gpt = resourceloader.transformer
text_data = resourceloader.text_data
block_size = gpt.config.block_size
encode, decode = resourceloader.load_tokenizer()
n_features, n_ffwd = autoencoder.encoder.weight.shape
html_out = os.path.join(os.path.dirname(os.path.abspath('.')), 'out', str(autoencoder_subdir))

X, _ = resourceloader.get_text_batch(num_contexts=num_contexts)

## glossary of variables
S = num_sampled_tokens
V = window_radius
M = num_contexts * S
W = 2 * V + 1 # window length
I = num_intervals
B = eval_batch_size

## create the main HTML page
create_main_html_page(n_features=n_features, dirpath=html_out)

# TODO: dynamically set n_features_per_phase and n_phases by reading off free memory in the system
## due to memory constraints, compute feature data in phases, processing n_features_per_phase features in each phase 
n_phases = n_features // n_features_per_phase + (n_features % n_features_per_phase !=0)
n_batches = num_contexts // B + (num_contexts % B != 0)
print(f"Will process features in {n_phases} phases. Each phase will have forward pass in {n_batches} batches")

for phase in range(n_phases): 
    H = n_features_per_phase if phase < n_phases - 1 else n_features - (phase * n_features_per_phase)
    # TODO: the calculation of H could probably be made better. Am I counting 1 extra? What about the case when 
    # n_features % n_features_per_phase == 0
    print(f'working on phase # {phase + 1}/{n_phases}: features # {phase * n_features_per_phase} through {phase * n_features_per_phase + H}')   
    ## compute and store feature activations # TODO: data_MW should be renamed to something more clear. 
    data_MW = TensorDict({
        "tokens": torch.zeros(M, W, dtype=torch.int32),
        "feature_acts_H": torch.zeros(M, W, H),
        }, batch_size=[M, W]
        )

    for iter in range(n_batches): 
        print(f"Computing feature activations for batch # {iter+1}/{n_batches} in phase # {phase + 1}/{n_phases}")
        # select text input for the batch
        x = X[iter * B: (iter + 1) * B].to(device)
        # compute MLP activations 
        _, _ = gpt(x)
        mlp_acts_BTF = gpt.mlp_activation_hooks[0]
        gpt.clear_mlp_activation_hooks() 
        # compute feature activations for features in this phase
        feature_acts_BTH = autoencoder.get_feature_activations(inputs=mlp_acts_BTF, 
                                                                start_idx=phase*H, 
                                                                end_idx=(phase+1)*H)
        # sample tokens from the context, and save feature activations and tokens for these tokens in data_MW.
        X_PW, feature_acts_PWH = select_context_windows(x, feature_acts_BTH, 
                                                        num_sampled_tokens=num_sampled_tokens, 
                                                        window_radius=window_radius, 
                                                        fn_seed=seed+iter) # P = B * S
        data_MW["tokens"][iter * B * num_sampled_tokens: (iter + 1) * B * num_sampled_tokens] = X_PW 
        data_MW["feature_acts_H"][iter * B * num_sampled_tokens: (iter + 1) * B * num_sampled_tokens] = feature_acts_PWH

        del mlp_acts_BTF, feature_acts_BTH, x, X_PW, feature_acts_PWH; gc.collect(); torch.cuda.empty_cache() 

    ## Get top k feature activations
    print(f'computing top k feature activations in phase # {phase + 1}/{n_phases}')
    _, topk_indices_kH = torch.topk(data_MW["feature_acts_H"][:, window_radius, :], k=num_top_activations, dim=0)
    # evaluate text windows and feature activations to topk_indices_kH to get texts and activations for top activations
    top_acts_data_kWH = TensorDict({
        "tokens": data_MW["tokens"][topk_indices_kH].transpose(dim0=1, dim1=2),
        "feature_acts": torch.stack([data_MW["feature_acts_H"][topk_indices_kH[:, i], :, i] for i in range(H)], dim=-1)
        }, batch_size=[num_top_activations, W, H])
        
    # TODO: is my definition of ultralow density neurons consistent with Anthropic's definition?
    # TODO: make sure there are no bugs in switch back and forth between feature id and h.
    # TODO: It seems that up until the computation of num_non_zero_acts,
    # we can use vectorization. It would look something like this.
    # mid_token_feature_acts_MH = data_MW["feature_acts_H"][:, window_radius, :]
    # num_nonzero_acts_MH = torch.count_nonzero(mid_token_feature_acts_MH, dim=1)
    # the sorting and sampling operations that follow seem harder to vectorize.
    # I wonder if there will be enough computational speed-up from vectorizing
    # the computation of num_nonzero_acts_MH as above.
    for h in range(H):
        
        feature_id = phase * n_features_per_phase + h
        ## check whether feature is alive, dead or ultralow density based on activations on sampled tokens
        curr_feature_acts_MW = data_MW["feature_acts_H"][:, :, h]
        mid_token_feature_acts_M = curr_feature_acts_MW[:, window_radius]
        num_nonzero_acts = torch.count_nonzero(mid_token_feature_acts_M)

        # if neuron is dead, write a dead neuron page
        if num_nonzero_acts == 0:
            write_dead_feature_page(feature_id=feature_id, dirpath=html_out)
            continue 

        ## make a histogram of non-zero activations
        act_density = torch.count_nonzero(curr_feature_acts_MW) / (M * W) * 100
        non_zero_acts = curr_feature_acts_MW[curr_feature_acts_MW !=0]
        make_histogram(activations=non_zero_acts, 
                        density=act_density, 
                        feature_id=feature_id,
                        dirpath=html_out)

        # if neuron has very few non-zero activations, consider it an ultralow density neurons
        if num_nonzero_acts < I * samples_per_interval:
            write_ultralow_density_feature_page(feature_id=feature_id, 
                                                decode=decode,
                                                top_acts_data=top_acts_data_kWH[:num_nonzero_acts, :, h],
                                                dirpath=html_out)
            continue

        ## sample I intervals of activations when feature is alive
        sorted_acts_M, sorted_indices_M = torch.sort(mid_token_feature_acts_M, descending=True)
        sampled_indices = torch.stack([j * num_nonzero_acts // I + 
                                       torch.randperm(num_nonzero_acts // I)[:samples_per_interval].sort()[0] 
                                        for j in range(I)], dim=0) # (I, SI)
        original_indices = sorted_indices_M[sampled_indices] # (I, SI)
        sampled_acts_data_IXW = TensorDict({
            "tokens": data_MW["tokens"][original_indices],
            "feature_acts": curr_feature_acts_MW[original_indices],
            }, batch_size=[I, samples_per_interval, W]) # (I, SI, W)

        # ## write feature page for an alive feature
        write_alive_feature_page(feature_id=feature_id, 
                                    decode=decode,
                                    top_acts_data=top_acts_data_kWH[:, :, h],
                                    sampled_acts_data = sampled_acts_data_IXW,
                                    dirpath=html_out)

    if phase == 1:
        break