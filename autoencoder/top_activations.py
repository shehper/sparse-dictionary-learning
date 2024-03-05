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
k = 10 # number of top activations for each feature; 20 in Anthropic's visualization
num_intervals = 11 # number of intervals to divide activations in; = 11 in Anthropic's work
interval_exs = 5 # number of examples to sample from each interval of activations 
modes_density_cutoff = 1e-3 # TODO: remove this; it is not being used anymor
publish_html = False
make_histogram = False

slice_fn = lambda data: data[iter * eval_batch_size: (iter + 1) * eval_batch_size]

def sample_tokens(*args, fn_seed=0):
    # given tensors each of shape (B, T, ...), return tensors on randomly selected tokens
    # and windows around them. shape of output: (B, M, W, ...)
    
    assert args and isinstance(args[0], torch.Tensor), "must provide at least one torch tensor as input"
    assert args[0].ndim >=2, "input tensor must at least have 2 dimensions"
    B, T = args[0].shape[:2]
    for tensor in args[1:]:
        assert tensor.shape[:2] == (B, T), "all tensors in input must have the same shape along the first two dimensions"

    torch.manual_seed(fn_seed)
    V = num_tokens_either_side # W = 2 * V + 1
    M = eval_tokens
    # select indices for tokens
    token_idx_BM = torch.randint(V, T - V, (B, M))
    # include windows
    window_idx_BMW = token_idx_BM.unsqueeze(-1) + torch.arange(-V, V + 1)
    # obtain batch indices
    batch_indices_BMW = torch.arange(B).view(-1, 1, 1).expand_as(window_idx_BMW)
    # print(args[1][batch_indices_BMW, window_idx_BMW, :])

    result_tensors = []
    for tensor in args:
        if tensor.ndim == 3:  # For (B, T, F) tensors
            sliced_tensor = tensor[batch_indices_BMW, window_idx_BMW, :]
        elif tensor.ndim == 2:  # For (B, T) tensors
            sliced_tensor = tensor[batch_indices_BMW, window_idx_BMW]
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
    autoencoder_ckpt_path = os.path.join(autoencoder_dir, autoencoder_subdir, 'ckpt.pt')
    autoencoder_ckpt = torch.load(autoencoder_ckpt_path, map_location=device)
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

    ## select X, Y from text data
    # as openwebtext and shakespeare_char are small datasets, use the entire text data. split it into len(text_data)//block_size contexts
    if dataset in ["openwebtext", "shakespeare_char"]:
        X_NT = torch.stack([torch.from_numpy((text_data[i:i+block_size]).astype(np.int64)) for i in range(len(text_data)//block_size)])
        # TODO: note that we don't need y_BT/y_NT unless we compute feature ablations
        # Y_NT = torch.stack([torch.from_numpy((text_data[i+1:i+1+block_size]).astype(np.int64)) for i in range(len(text_data)//block_size)])
        num_contexts = X_NT.shape[0] # overwrite num_contexts = N
    else:
        raise NotImplementedError("""if the text dataset is too large, such as The Pile, you may not evaluate on the whole dataset.
                                    In this case, use get_text_batch function to randomly select num_contexts contexts""") # TODO

    ## compute and store MLP activations 
    data_NMW = TensorDict({
        "mlp_acts": torch.empty(num_contexts, eval_tokens, 2 * num_tokens_either_side + 1, n_ffwd),
        "tokens": torch.empty(num_contexts, eval_tokens, 2 * num_tokens_either_side + 1),
        "feature_acts": torch.empty(num_contexts, eval_tokens, 2 * num_tokens_either_side + 1),
    }, batch_size=[num_contexts, eval_tokens, 2 * num_tokens_either_side + 1]
    )
    # TODO: only process one feature at a time. There will be a trade-off (at least for now) 
    # between how many features to process versus
    # how much data to use because of memory and storage constraints. 

    # mlp_acts_NMWF = torch.empty(num_contexts, eval_tokens, 2 * num_tokens_either_side + 1, n_ffwd)
    num_batches = num_contexts // eval_batch_size
    print(f"We will compute MLP activations in {num_batches} batches")
    for iter in range(num_batches):    
        print(f"Computing MLP activations for batch # {iter+1}/{num_batches}")
        
        # select text input for the batch
        x_BT = slice_fn(X_NT).pin_memory().to(device, non_blocking=True) if device_type == 'cuda' else slice_fn(X_NT).to(device)
        
        # compute MLP activations 
        batch_mlp_acts_BTF = gpt.get_mlp_acts(x_BT) # (B, T, F) # TODO: Learn to use hooks instead? 
        
        # sample tokens and store MLP activations only for the sampled tokens and windows around them
        batch_mlp_acts_BMWF, batch_tokens_BMW = sample_tokens(batch_mlp_acts_BTF, x_BT, fn_seed=seed+iter)
        data_NMW["mlp_acts"][iter * eval_batch_size: (iter + 1) * eval_batch_size] = batch_mlp_acts_BMWF.to(device='cpu')
        data_NMW["tokens"][iter * eval_batch_size: (iter + 1) * eval_batch_size] = batch_tokens_BMW
        
    print(data_NMW["tokens"][-75:-70])
    print(data_NMW["mlp_acts"][-75:-70])
    print(num_contexts)
    print(torch.count_nonzero(data_NMW["tokens"]))
    # TODO: can't figure out why some of the tokens stay zero. hmm. 
    # TODO: also figure out what changes to make in order to make this work with cuda. 
    raise 

    # TODO: p should probably be called W instead as it is the length of the window
    # TODO: n should be called M -- this is the number of tokens per eval context that we will be computing 

#    feature_infos = {i: [] for i in range(n_features)}
    # TODO: compute the largest batch size for which I should be able to evaluate

    # for each feature, we want at the end, 
    # a tensor of k windows
    # for each window, we want all token ids and feature activation values
    # so (k, W, 2) 
    # and a tensor of shape
    # (num_intervals, interval_exs, W, 2) 
    # These could be obtained from 
    # 
    print(f"Now we will compute feature activations")
    for feature_id in range(n_features):
        feature_activations = torch.tensor() # TODO: what should be the shape?
        # num_contexts, eval_tokens, 2 * num_tokens_on_each_side + 1
        for iter in range(num_batches):
            print(f"Computing feature activations for feature # {feature_id} in batch # {iter+1}/{num_batches}")

            # select batch of mlp activations
            if device_type == 'cuda':
                batch_mlp_activations_BTpF = slice_fn(mlp_acts_NtpF).pin_memory().to(device, non_blocking=True) 
            else:
                batch_mlp_activations_BTpF = slice_fn(mlp_acts_NtpF).to(device) # (B, T, p, F)

            with torch.no_grad():
                output = autoencoder(batch_mlp_activations)
            
            batch_f = output['f'].to('cpu') # (eval_batch_size, block_size, n_features)
            batch_token_indices = slice_fn(token_indices) # (eval_batch_size, tokens_per_eval_context)
            batch_contexts = slice_fn(X) # (eval_batch_size, block_size)

        # TODO: this is picking different subset of tokens for different features. 
        # Instead replace the outer for loop with 
        # sample_idx = torch.randint(context_length, block_size - context_length, (eval_batch_size, tokens_per_eval_context))
        # batch_f_subset = torch.gather(batch_f, 1, sample_idx.unsqueeze(dim=2).expand(-1, -1, n_features))
            
        for i in range(n_features):
            curr_f = batch_f[:, :, i] # (eval_batch_size, block_size)
            # now pick 10 random tokens
            sample_idx = torch.randint(context_length, block_size - context_length, (eval_batch_size, tokens_per_eval_context)) # (eval_batch_size, tokens_per_eval_context)
            # evaluate curr_f on these tokens
            curr_f_subset = torch.gather(curr_f, 1, sample_idx) # (eval_batch_size, tokens_per_eval_context)
            # for each token in sample_idx, get the token and feature activation for that token 
            for k in range(eval_batch_size):
                for m in range(tokens_per_eval_context):
                    if curr_f_subset[k, m] != 0:
                        sample_c_idx = [l for l in range(sample_idx[k, m] - context_length, sample_idx[k, m] + context_length + 1)]
                        context = batch_contexts[k, sample_c_idx]
                        f_acts = batch_f[k, sample_c_idx, i]
                        context_acts = [(s, t) for s, t in zip(context, f_acts)]
                        feature_infos[i] += [(curr_f_subset[k, m].item(), context_acts)]


    if publish_html:
        ## load tokenizer used to train the gpt model
        # look for the meta pickle in case it is available in the dataset folder
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

        ## import functions needed to write html files
        from write_html import main_page, tooltip_css, feature_page
        
        # create a directory to store pages
        os.makedirs(os.path.join(autoencoder_dir, autoencoder_subdir, 'pages'), exist_ok=True)
        # write a helper css file tooltip.css in autoencoder_subdir
        with open(os.path.join(autoencoder_dir, autoencoder_subdir, f'tooltip.css'), 'w') as file:
            file.write(tooltip_css()) 
        # write the main page for html
        with open(os.path.join(autoencoder_dir, autoencoder_subdir, 'main.html'), 'w') as file:
            file.write(main_page(n_features))
        print(f'wrote tooltip.css and main.html in {os.path.join(autoencoder_dir, autoencoder_subdir)}')

        for i, feature_info in enumerate(feature_infos):
            if i % 100 == 0:
                print(f'working on neurons {i} through {i+99}')
            with open(os.path.join(autoencoder_dir, autoencoder_subdir, 'pages', f'page{i}.html'), 'w') as file:
                file.write(feature_page(i, feature_infos[i], decode, make_histogram, k, num_intervals, interval_exs, autoencoder_dir, autoencoder_subdir))  

