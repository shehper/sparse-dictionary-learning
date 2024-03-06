"""
Load a trained autoencoder model to compute its top activations

Run on a macbook on a Shakespeare dataset as 
python top_activations.py --device=cpu --dataset=shakespeare_char --gpt_dir=out-shakespeare-char --device=cpu --num_contexts=1000 --autoencoder_subdir=1704914564.90-autoencoder-shakespeare_char
"""

import torch
from tensordict import TensorDict 
import os
import numpy as np
import pickle # needed to load meta.pkl
import tiktoken # needed to decode contexts to text
import sys 
from autoencoder import AutoEncoder
from write_html import *

## Add path to the transformer subdirectory as it contains GPT class in model.py
sys.path.insert(0, '../transformer')
from model import GPTConfig, GPT

# hyperparameters 
device = 'cuda' # change it to cpu
seed = 1442
dataset = 'openwebtext' 
gpt_dir = 'out' 
autoencoder_dir = 'out_autoencoder' # directory containing weights of various trained autoencoder models
autoencoder_subdir = '' # subdirectory containing the specific model to consider
eval_batch_size = 156 # batch size for computing reconstruction nll # TODO: this should have a different name. # B
num_contexts = 10000 # 10 million in anthropic paper; but we will choose the entire dataset as our dataset is small # N
eval_tokens = 10 # same as Anthropic's paper; number of tokens in each context on which feature activations will be computed # M
num_tokens_either_side = 4 # number of tokens to print/save on either side of the token with feature activation. 
# let 2 * num_tokens_either_side + 1 be denoted by W.
n_features_per_phase = 20 # due to memory constraints, it's useful to process features in phases. 
k = 10 # number of top activations for each feature; 20 in Anthropic's visualization
n_intervals = 12 # number of intervals to divide activations in; = 12 in Anthropic's work
n_exs_per_interval = 5 # number of examples to sample from each interval of activations 
modes_density_cutoff = 1e-3 # TODO: remove this; it is not being used anymor

def sample_tokens(*args, fn_seed=0, U=eval_tokens, V=num_tokens_either_side):
    # given tensors each of shape (B, T, ...), return tensors on randomly selected tokens
    # and windows around them. shape of output: (B * U, W, ...)
    
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
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    assert config['autoencoder_subdir'], "autoencoder_subdir must be provided to load a trained autoencoder model"

    # variables that depend on input parameters
    config['device_type'] = device_type = 'cuda' if 'cuda' in device else 'cpu'
        
    torch.manual_seed(seed)

    # load autoencoder model weights
    autoencoder_path = os.path.join(autoencoder_dir, autoencoder_subdir)
    autoencoder_ckpt = torch.load(os.path.join(autoencoder_path, 'ckpt.pt'), map_location=device)
    state_dict = autoencoder_ckpt['autoencoder']
    n_features, n_ffwd = state_dict['enc.weight'].shape # H, F
    l1_coeff = autoencoder_ckpt['config']['l1_coeff']
    autoencoder = AutoEncoder(n_ffwd, n_features, lam=l1_coeff).to(device)
    autoencoder.load_state_dict(state_dict)

    ## load tokenized text data
    current_dir = os.path.abspath('.')
    data_dir = os.path.join(os.path.dirname(current_dir), 'transformer', 'data', dataset)
    text_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

    ## load GPT model --- we need it to compute reconstruction nll and nll score
    gpt_ckpt_path = os.path.join(os.path.dirname(current_dir), 'transformer', gpt_dir, 'ckpt.pt')
    gpt_ckpt = torch.load(gpt_ckpt_path, map_location=device)
    gptconf = GPTConfig(**gpt_ckpt['model_args'])
    gpt = GPT(gptconf)
    state_dict = gpt_ckpt['model']
    compile = False # TODO: why do this?
    unwanted_prefix = '_orig_mod.' # TODO: why do this and the next three lines?
    for key, val in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    gpt.load_state_dict(state_dict)
    gpt.eval()
    gpt.to(device)
    if compile:
        gpt = torch.compile(gpt) # requires PyTorch 2.0 (optional)
    config['block_size'] = block_size = gpt.config.block_size # T

    ## load tokenizer
    load_meta = False
    meta_path = os.path.join(os.path.dirname(current_dir), 'transformer', 'data', gpt_ckpt['config']['dataset'], 'meta.pkl')
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

    ## select X, Y from text data
    # as openwebtext and shakespeare_char are small datasets, use the entire text data. split it into len(text_data)//block_size contexts
    T = block_size
    if dataset in ["openwebtext", "shakespeare_char"]:
        X_NT = torch.stack([torch.from_numpy((text_data[i*T: (i+1)*T]).astype(np.int64)) for i in range(len(text_data)//T)])
        # TODO: note that we don't need y_BT/y_NT unless we compute feature ablations which we do plan to compute 
        # Y_NT = torch.stack([torch.from_numpy((text_data[i+1:i+1+block_size]).astype(np.int64)) for i in range(len(text_data)//block_size)])
        num_contexts = N = X_NT.shape[0] # overwrite num_contexts = N
    else:
        raise NotImplementedError("""if the text dataset is too large, such as The Pile, you may not evaluate on the whole dataset.
                                    In this case, use get_text_batch function to randomly select num_contexts contexts""") # TODO

    ## create the main HTML page
    create_main_html_page(n_features=n_features, dirpath=autoencoder_path)

    ## glossary of variables
    T = block_size
    N = num_contexts
    U = eval_tokens
    V = num_tokens_either_side
    M = N * U
    W = 2 * V + 1 # window length
    I = n_intervals
    X = n_exs_per_interval
    B = eval_batch_size

    # TODO: dynamically set n_features_per_phase and n_phases by reading off free memory in the system
    # TODO: ensure compatability with CUDA
    # due to memory constraints, compute feature data in phases, processing n_features_per_phase features in each phase 
    n_phases = n_features // n_features_per_phase + (n_features % n_features_per_phase !=0)
    n_batches = N // B + (N % B != 0)
    print(f"Will process features in {n_phases} phases. Each phase will have forward pass in {n_batches} batches")

    for phase in range(n_phases): 
        H = n_features_per_phase if phase < n_phases - 1 else n_features - (phase * n_features_per_phase)
        # TODO: the calculation of H could probably be made better. Am I counting 1 extra? What about the case when 
        # n_features % n_features_per_phase == 0
        print(f'working on phase # {phase + 1}/{n_phases}: features # {phase * n_features_per_phase} through {phase * n_features_per_phase + H}')   
        ## compute and store feature activations 
        data_MW = TensorDict({
            "tokens": torch.empty(M, W),
            "feature_acts_H": torch.empty(M, W, H),
            }, batch_size=[M, W]
            )

        for iter in range(n_batches): 
            print(f"Computing feature activations for batch # {iter+1}/{n_batches} in phase # {phase + 1}/{n_phases}")
            # select text input for the batch
            if device_type == 'cuda':
                X_BT = X_NT[iter * B: (iter + 1) * B].pin_memory().to(device, non_blocking=True) 
            else:
                X_BT = X_NT[iter * B: (iter + 1) * B].to(device)
            # compute MLP activations 
            mlp_acts_BTF = gpt.get_mlp_acts(X_BT) # TODO: Learn to use hooks instead? 
            # compute feature activations # TODO: evaluate forward pass on only the portion of the autoencoder relevant to current features
            feature_acts_BTH = autoencoder.get_feature_acts(mlp_acts_BTF)[:, :, phase * H: (phase + 1) * H]
            # sample tokens from the context, and save feature activations and tokens for these tokens in data_MW.
            X_PW, feature_acts_PWH = sample_tokens(X_BT, feature_acts_BTH, fn_seed=seed+iter) # P = B * U
            data_MW["tokens"][iter * B * U: (iter + 1) * B * U] = X_PW
            data_MW["feature_acts_H"][iter * B * U: (iter + 1) * B * U] = feature_acts_PWH

        ## Get top k feature activations
        print(f'computing top k feature activations in phase # {phase + 1}/{n_phases}')
        values_kH, indices_kH = torch.topk(data_MW["feature_acts_H"][:, num_tokens_either_side, :], k=k, dim=0)
        top_feature_acts_kWH = torch.stack([data_MW["feature_acts_H"][indices_kH[:, i], :, i] for i in range(H)], dim=-1) 
        top_windows_kWH = data_MW["tokens"][indices_kH].transpose(dim0=1, dim1=2)
        assert top_feature_acts_kWH.shape == (k, W, H), "top feature activations do not have the right shape"
        assert top_windows_kWH.shape == (k, W, H), "windows for top feature activations do not have the right shape"
    
        # TODO: save histograms in a separate directory. 
        # TODO: consider making a histogram in plotly instead?
        # TODO: can we do this work in parallel? perhaps using Ray?
        for h in range(H):
            
            feature_id = phase * n_features_per_phase + h
            ## check whether feature is alive, dead or ultralow density based on activations on sampled tokens
            curr_feature_acts_MW = data_MW["feature_acts_H"][:, :, h]
            mid_token_feature_acts_M = curr_feature_acts_MW[:, num_tokens_either_side]
            num_non_zero_acts = torch.count_nonzero(mid_token_feature_acts_M)
            if num_non_zero_acts < I * X:
                if num_non_zero_acts == 0:
                    # print(f'dead neuron with non zero activation count: {num_non_zero_acts}')
                    write_dead_feature_page(feature_id=feature_id, dirpath=autoencoder_path)
                else:
                    # print(f'ultra low density with non zero activation count: {num_non_zero_acts}')
                    write_ultralow_density_feature_page(feature_id=feature_id, dirpath=autoencoder_path)
                continue

            ## sample I intervals of activations as feature is alive
            sorted_acts_M, sorted_indices_M = torch.sort(mid_token_feature_acts_M, descending=True)
            sampled_indices_IX = torch.stack([j * num_non_zero_acts // I + torch.randperm(num_non_zero_acts // I)[:X].sort()[0] for j in range(I)], dim=0)
            original_indices_IX = sorted_indices_M[sampled_indices_IX]
            sampled_acts_IXW = curr_feature_acts_MW[original_indices_IX]
            sampled_tokens_IXW = data_MW["tokens"][original_indices_IX]

            ## make a histogram of non-zero activations
            # TODO: consider using Plotly for these images. -- If we save an image, can we still hover over it?
            act_density = torch.count_nonzero(curr_feature_acts_MW) / (M * W) * 100
            non_zero_acts = curr_feature_acts_MW[curr_feature_acts_MW !=0]
            make_histogram(activations=non_zero_acts, 
                           density=act_density, 
                           feature_id=feature_id,
                           dirpath=autoencoder_path)

            # ## write feature page
            write_alive_feature_page(feature_id=feature_id, dirpath=autoencoder_path)

        break


            ## add sampled activations to the feature page



        # plot a histogram with non_zero_acts



        # TODO: replace torch.randint by torch.randperm so that we can sample without replacement?


        # print(f'printing subsample activations for feature # {feature}')
        # for j in range(12):
        #     print(f'printing subsample interval # {j}')
        #     for k in range(5):
        #         print(f'example # {k + 1}: {decode(sampled_tokens_IXW[j, k].tolist())}, \
        #               activations: {sampled_acts_IXW[j, k]}')



    ## USING FOR LOOP
    # for each feature:
    #   sort all the activations in a descending order
    #   count the number of non-zero elements
    #   divide the number of non-zero elements by 11
    #   in each split induced by this division, pick 5 elements at random
    #   but how do we use it to get indices for the rows of tokens?

    # remember that feature_acts has shape MWH
    # tokens has shape MW
    # TODO: how do I make the two work together?
    # well, it might be easier to do it one feature at a time
    # for each feature, look at the middle point of the window
    # and sort

    #         assert curr_feature_acts_MW.shape == (M, W)
    #         assert mid_token_feature_acts_M.shape == (M, )
    #         assert sorted_acts_M.shape == (M, )
    #         assert sorted_indices_M.shape == (M, )
    #         assert num_non_zero_acts.shape == ()
    #         assert sampled_indices_IX.shape == (I, X)
    #         assert original_indices_IX.shape == (I, X)
    #         assert sampled_acts_IXW.shape == (I, X, W)
    #         assert sampled_tokens_IXW.shape == (I, X, W)
    # # 

    # for i in range(n_features_per_phase):
    #     print(f'printing top activations for feature # {i}')
    #     for j in range(k):
    #         print(f'activation # {j}: {decode(topk_windows_kWH[j, :, i].tolist())}; mid value: {top_feature_acts_kWH[j, num_tokens_either_side, i]}, index: {indices_kH[j, i]}')



    # List of things to record: 
    # 1. neuron alignment
    # 2. correlated neurons
    # 3. subsample activations
    # 4. histogram
    # 5. feature ablations (positive logits, negative logits, etc.)

    # #1 and #2 can be done later. 
    # #3, # 4 and # 5 are more important.
    # let's think about #3 first.



    #    activation values: {top_feature_acts_kWH[j, :, i]}, \
    # TODO: why does the same token and context appear many a times but with different activation values? 
    # make sure the algebra for calculation of top k feature activations and windows is correct. 

    # for i in range(n_features):
    #     if topk_feature_acts[0] =
    #     print(f'printing top activations for feature # {i}')
    # TODO: should redefine a variable named 'window_size'

    # TODO: I think that 
    # data_MW["tokens"][topk_indices].transpose(dim0=1, dim1=2) will give (k, W, H)
    # of k top windows for all features
    # torch.stack([data_MW["feature_acts"][indices[:, i], :, i] for i in range(a.shape[-1])], dim=-1) 
    # will also give a tensor of shape (k, W, H).
    # this tensor will have feature activations for topk windows of all H features.
    ## 

    ## subsample intervals
    #for feature_id in range(n_features):
        # perhaps sort all the activations by the middle index
        # now sample at intervals

    # # plot a histogram
    # for feature_id in range(n_features):
    #     num_nonzero_acts = torch.count_nonzero(data_MW["feature_acts"][:, :, feature_id])
    #     feature_density = num_nonzero_acts / (data_MW["tokens"].numel())
    #     # TODO: Include code to write feature page and plot a histogram

    # raise

    # Logic so far and what's ahead
    # we have contexts on which we compute MLP activations and feature activations
    # but we save tokens and feature activations for only a subset of tokens and windows around them
    # (B, M, W, H) 
    # TODO: Can one flatten the first two dimensions?
    # now we want to sort by the value of feature activations at the middle token of the windows
    # i.e. for each fixed value along the last dimension, we want to find top k elements along the first two dims
    # i.e. a tensor of shape (k, H)
    # but k 
    # now fix the value along the last dimension, say 0
    # and iterate over the 0th dimension,
    # 

    # print(torch.topk(data_MW["feature_acts"], k=5, dim=-1))
    # print(data_NMW["tokens"])
    # print(data_NMW["feature_acts"])
    # print(data_NMW["tokens"].shape)
    # print(data_NMW["feature_acts"].shape)
    # print(data_NMW["feature_acts"].numel())
    # print(torch.count_nonzero(data_NMW["feature_acts"]))
    # print(torch.count_nonzero(data_NMW["feature_acts"])/ data_NMW["feature_acts"].numel())
    # TODO: can't figure out why some of the tokens stay zero. hmm. 
    # TODO: also figure out what changes to make in order to make this work with cuda. 


 