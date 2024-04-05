"""
Make a feature browser for a trained autoencoder model.
Run on a Macbook as
python build_website.py --device=cpu --dataset=shakespeare_char --gpt_ckpt_dir=out_sc_1_2_32 --autoencoder_subdir=1712254759.95
"""

import torch
from tensordict import TensorDict 
import os
import numpy as np
import sys 
import gc
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
num_contexts = 10000 # N
eval_batch_size = 156 # batch size for computing reconstruction nll # TODO: this should have a different name. # B
# feature page hyperparameters
num_sampled_tokens = 10 # number of tokens in each context on which feature activations will be computed # M
window_radius = 4 # number of tokens to print on either side of a sampled token.. 
k = 10 # number of top activations for each feature # TODO: should probably be called num_top_acts = 
n_intervals = 12 # number of intervals to divide activations in; = 12 in Anthropic's work
n_exs_per_interval = 5 # number of examples to sample from each interval of activations 
n_features_per_phase = 20 # due to memory constraints, it's useful to process features in phases.
# system
device = 'cuda' # change it to cpu
# reproducibility
seed = 1442

def select_context_windows(*args, num_sampled_tokens, window_radius, fn_seed=0):
    """
    Select windows of tokens around randomly sampled tokens from input tensors.

    Given tensors each of shape (B, T, ...), this function returns tensors containing
    windows around randomly selected tokens. The shape of the output is (B * U, W, ...),
    where U is the number of tokens in each context to evaluate, and W is the window size
    (including the token itself and tokens on either side).

    Parameters:
    - args: Variable number of tensor arguments, each of shape (B, T, ...)
    - num_sampled_tokens (int): The number of tokens in each context on which to evaluate
    - window_radius (int): The number of tokens on either side of the sampled token
    - fn_seed (int, optional): Seed for random number generator, default is 0

    Returns:
    - A list of tensors, each of shape (B * U, W, ...), where U is `num_sampled_tokens` and W is
      the window size calculated as 2 * `window_radius` + 1.

    Raises:
    - AssertionError: If no tensors are provided, or if the tensors do not have the required shape.

    Example usage:
    ```
    tensor1 = torch.randn(10, 20, 30)  # Example tensor
    windows = select_context_windows(tensor1, num_sampled_tokens=5, window_radius=2)
    ```
    """

    U, V = num_sampled_tokens, window_radius
    
    assert args and isinstance(args[0], torch.Tensor), "must provide at least one torch tensor as input"
    assert args[0].ndim >=2, "input tensor must at least have 2 dimensions"
    B, T = args[0].shape[:2]
    for tensor in args[1:]:
        assert tensor.shape[:2] == (B, T), "all tensors in input must have the same shape along the first two dimensions"

    # window size
    W = 2 * V + 1
    torch.manual_seed(fn_seed)
    # select indices for tokens --- pick M elements without replacement in each batch 
    token_idx_BU = torch.stack([V + torch.randperm(T - 2*V)[:U] for _ in range(B)], dim=0)
    # include windows
    window_idx_BUW = token_idx_BU.unsqueeze(-1) + torch.arange(-V, V + 1)
    # obtain batch indices
    batch_indices_BUW = torch.arange(B).view(-1, 1, 1).expand_as(window_idx_BUW)

    result_tensors = []
    for tensor in args:
        if tensor.ndim == 3:  # For (B, T, H) tensors such as MLP activations
            H = tensor.shape[2] # number of features / hidden dimension of autoencoder, hence abbreviated to H
            sliced_tensor = tensor[batch_indices_BUW, window_idx_BUW, :].view(-1, W, H)
        elif tensor.ndim == 2:  # For (B, T) tensors such as inputs to Transformer
            sliced_tensor = tensor[batch_indices_BUW, window_idx_BUW].view(-1, W)
        else:
            raise ValueError("Tensor dimensions not supported. Only 2D and 3D tensors are allowed.")
        result_tensors.append(sliced_tensor)

    return result_tensors

if __name__ == '__main__':

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
    
    ## select X, Y from text data
    # TODO: use resourceloader.get_text_batch here. 
    # X = resourceloader.get_text_batch(num_contexts=num_contexts)
    T = block_size
    N = num_contexts
    # if number of contexts is too large (larger than length of data//block size), may as well use the entire dataset
    if len(text_data) < N * T:
        N = num_contexts = len(text_data)//block_size # overwrite N
        ix = torch.tensor([i*T for i in range(N)])
    else:
        ix = torch.randint(len(text_data) - T, (N,))
    X_NT = torch.stack([torch.from_numpy((text_data[i: i+T]).astype(np.int32)) for i in ix])

    ## glossary of variables
    U = num_sampled_tokens
    V = window_radius
    M = N * U
    W = 2 * V + 1 # window length
    I = n_intervals
    X = n_exs_per_interval
    B = eval_batch_size
    
    ## create the main HTML page
    create_main_html_page(n_features=n_features, dirpath=html_out)

    # TODO: dynamically set n_features_per_phase and n_phases by reading off free memory in the system
    ## due to memory constraints, compute feature data in phases, processing n_features_per_phase features in each phase 
    n_phases = n_features // n_features_per_phase + (n_features % n_features_per_phase !=0)
    n_batches = N // B + (N % B != 0)
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
            X_BT = X_NT[iter * B: (iter + 1) * B].to(device)
            # compute MLP activations 
            _, _ = gpt(X_BT)
            mlp_acts_BTF = gpt.mlp_activation_hooks[0]
            gpt.clear_mlp_activation_hooks() 
            # compute feature activations for features in this phase
            feature_acts_BTH = autoencoder.get_feature_activations(inputs=mlp_acts_BTF, 
                                                                   start_idx=phase*H, 
                                                                   end_idx=(phase+1)*H)
            # sample tokens from the context, and save feature activations and tokens for these tokens in data_MW.
            X_PW, feature_acts_PWH = select_context_windows(X_BT, feature_acts_BTH, num_sampled_tokens=U, window_radius=V, fn_seed=seed+iter) # P = B * U
            data_MW["tokens"][iter * B * U: (iter + 1) * B * U] = X_PW 
            data_MW["feature_acts_H"][iter * B * U: (iter + 1) * B * U] = feature_acts_PWH

            del mlp_acts_BTF, feature_acts_BTH, X_BT, X_PW, feature_acts_PWH; gc.collect(); torch.cuda.empty_cache() 

        ## Get top k feature activations
        print(f'computing top k feature activations in phase # {phase + 1}/{n_phases}')
        _, topk_indices_kH = torch.topk(data_MW["feature_acts_H"][:, window_radius, :], k=k, dim=0)
        # evaluate text windows and feature activations to topk_indices_kH to get texts and activations for top activations
        top_acts_data_kWH = TensorDict({
            "tokens": data_MW["tokens"][topk_indices_kH].transpose(dim0=1, dim1=2),
            "feature_acts": torch.stack([data_MW["feature_acts_H"][topk_indices_kH[:, i], :, i] for i in range(H)], dim=-1)
            }, batch_size=[k, W, H])
            
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
            if num_nonzero_acts < I * X:
                write_ultralow_density_feature_page(feature_id=feature_id, 
                                                    decode=decode,
                                                    top_acts_data=top_acts_data_kWH[:num_nonzero_acts, :, h],
                                                    dirpath=html_out)
                continue

            ## sample I intervals of activations when feature is alive
            sorted_acts_M, sorted_indices_M = torch.sort(mid_token_feature_acts_M, descending=True)
            sampled_indices_IX = torch.stack([j * num_nonzero_acts // I + torch.randperm(num_nonzero_acts // I)[:X].sort()[0] for j in range(I)], dim=0)
            original_indices_IX = sorted_indices_M[sampled_indices_IX]
            sampled_acts_data_IXW = TensorDict({
                "tokens": data_MW["tokens"][original_indices_IX],
                "feature_acts": curr_feature_acts_MW[original_indices_IX],
                }, batch_size=[I, X, W])

            # ## write feature page for an alive feature
            write_alive_feature_page(feature_id=feature_id, 
                                     decode=decode,
                                     top_acts_data=top_acts_data_kWH[:, :, h],
                                     sampled_acts_data = sampled_acts_data_IXW,
                                     dirpath=html_out)

        if phase == 1:
            break