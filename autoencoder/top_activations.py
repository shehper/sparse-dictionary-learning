"""
Load a trained autoencoder model to compute its top activations

Run on a macbook on a Shakespeare dataset as 
python top_activations.py --device=cpu --dataset=shakespeare_char --gpt_dir=out-shakespeare-char --device=cpu --eval_contexts=1000 --autoencoder_subdir=1704914564.90-autoencoder-shakespeare_char
"""

import torch 
import os
import numpy as np
import pickle # needed to load meta.pkl
import tiktoken # needed to decode contexts to text
import sys 
from autoencoder import AutoEncoder
from train import get_text_batch

## Add path to the transformer subdirectory as it contains GPT class in model.py
sys.path.insert(0, '../transformer')
from model import GPTConfig, GPT

# hyperparameters --- same as train_sae.py except a few maybe # TODO: do I need all of them?
device = 'cuda' # change it to cpu
seed = 1442
dataset = 'openwebtext' 
gpt_dir = 'out' 
eval_batch_size = 16 # batch size for computing reconstruction nll 
eval_contexts = 10000 # 10 million in anthropic paper; but we can choose 1 million as OWT dataset is smaller
tokens_per_eval_context = 10 # same as anthropic paper
autoencoder_dir = 'out_autoencoder' # directory containing weights of various trained autoencoder models
autoencoder_subdir = '' # subdirectory containing the specific model to consider
context_length = 4 # number of tokens to print/save on either side of the token with feature activation. 
k = 20 # number of top activations
num_intervals = 11 # number of intervals to divide activations in; = 11 in Anthropic's work
interval_exs = 5 # number of examples to sample from each interval of activations 
modes_density_cutoff = 1e-3 # TODO: remove this; it is not being used anymor
publish_html = False
make_histogram = False
k = 15 # number of top activations
num_intervals = 11 # number of intervals to divide activations in; = 11 in Anthropic's work
interval_exs = 5 # number of examples to sample from each interval of activations 

slice_fn = lambda storage: storage[iter * eval_batch_size: (iter + 1) * eval_batch_size]

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
    n_features, n_ffwd = state_dict['enc.weight'].shape
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
    config['block_size'] = block_size = gpt.config.block_size

    ## Get text data for evaluation 
    X, Y = get_text_batch(text_data, block_size=block_size, batch_size=eval_contexts) # (eval_contexts, block_size) 

    # Compute and store MLP activations on evaluation text data
    mlp_activations_storage = torch.tensor([], dtype=torch.float16) # TODO: Maybe this should be pre-defined as zeros
    
    ## start evaluation
    num_eval_batches = eval_contexts // eval_batch_size
    for iter in range(num_eval_batches):    
        
        print(f'iter = {iter}/{num_eval_batches-1} in computation of mlp_acts for eval data')

        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = slice_fn(X).pin_memory().to(device, non_blocking=True) if device_type == 'cuda' else slice_fn(X).to(device)
        y = slice_fn(Y).pin_memory().to(device, non_blocking=True) if device_type == 'cuda' else slice_fn(Y).to(device)

        _, mlp_activations, batch_loss, batch_ablated_loss = gpt.forward_with_and_without_mlp(x, y)    
        mlp_activations_storage = torch.cat([mlp_activations_storage, mlp_activations.to(dtype=torch.float16, device='cpu')])

    # sample indices of tokens from each context where feature activations will be computed.
    token_indices = torch.randint(context_length, block_size - context_length, (eval_contexts, tokens_per_eval_context)) # (eval_contexts, tokens_per_eval_context)
    # TODO: This assumes that the tokens chosen for studying top activations are not on the extreme ends of any text in the context window of gpt
    # Can the generality we lose this way significantly affect our analysis? Put another way, could there be some interesting behavior on the edges of the context window?

    # The logic should be as follows:
    # For each context we should compute feature activations as in batch_f below
    # Restrict to token_indices #TODO: is this really necessary or should we consider all tokens in each context? What difference does this make conceptually?
    # Keep a dictionary feature_infos = {i: [] for i in range(n_features)]
    # Whenever a feature activation is non-zero, add it to the list as (feature_activation, token with context around it)
    # At the very end, sample 
    # TODO: I think one thing that I should also try to keep track of is the location of the start of each context in train.bin
    # Why? Well, If I intend to compute correlation between contexts of top activations, this might be needed. 
    # Otherwise, I could just compute it from my list feature_infos[i].
    # By the way, I should look at MMCS/cosine similarity in the paper. They might already be doing what I am intending to do. 

    feature_infos = {i: [] for i in range(n_features)}

    for iter in range(num_eval_batches):
        print(f'{iter}/{num_eval_batches-1} for evaluation')

        # select batch of mlp activations, residual stream and y 
        if device_type == 'cuda':
            batch_mlp_activations = slice_fn(mlp_activations_storage).pin_memory().to(device, non_blocking=True) 
        else:
            batch_mlp_activations = slice_fn(mlp_activations_storage).to(device) # (eval_batch_size, block_size, n_ffwd)

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

    #with open(os.path.join(autoencoder_dir, autoencoder_subdir, 'feature_infos.pkl'), 'wb') as f:
    #    pickle.dump(feature_infos, f)
    #print(f'saved feature_infos.pkl in {os.path.join(autoencoder_dir, autoencoder_subdir)}')

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

