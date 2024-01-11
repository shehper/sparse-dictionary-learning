"""
Load a trained autoencoder model to compute its top activations

Run on a macbook on a Shakespeare dataset as 
python top_activations.py --device=cpu --dataset=shakespeare_char --gpt_dir=out-shakespeare-char --device=cpu --eval_contexts=1000 --autoencoder_subdir=/subdirectory/of/out_autoencoder/containing_model_ckpt
"""

import torch 
import os
import numpy as np
import pickle # needed to load meta.pkl
import tiktoken # needed to decode contexts to text
import sys 
import io
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
length_context_on_each_side = 4 # number of tokens to print/save on either side of the token with top feature activation
k = 5 # number of top activations
modes_density_cutoff = 1e-3
publish_html = False

slice_fn = lambda storage: storage[iter * eval_batch_size: (iter + 1) * eval_batch_size]

def top_activations_and_tokens(f_subset, token_indices, contexts, context_on_each_side, k):
    """ 
        computes top feature activations for each feature and the corresponding tokens (with context around them)
    input: f_subset of shape (eval_batch_size, tokens_per_eval_context, n_features)
              token_indices of shape  (eval_batch_size, tokens_per_eval_context))
              contexts of shape (eval_batch_size, block_size)
              context_on_each_side: an int, must satisfy (2 * context_on_each_side + 1) < block_size
              k: an int, the number of top activation values to compute and return, must be < eval_batch_size * tokens_per_eval_context
    returns: 
              top_values: a tensor of shape (k, n_features)
              top_tokens_with_context: a tensor of shape (k, n_features, 2 * context_on_each_side + 1)
              """

    assert f_subset.shape == (eval_batch_size, tokens_per_eval_context, n_features)
    assert token_indices.shape == (eval_batch_size, tokens_per_eval_context)
    assert contexts.shape == (eval_batch_size, block_size)
    assert (2 * context_on_each_side + 1) < block_size, "this code will give errors if length_context_on_each_side is too long compared to block_size"
    assert k < eval_batch_size * tokens_per_eval_context, "number of top values to choose cannot be bigger than the number of tokens in each batch"

    flattened_f = f_subset.view(-1, f_subset.shape[-1]) # (m * n, p)
    top_values, flattened_indices = torch.topk(flattened_f, k=k, dim=0) # (k, p) 
    unflattened_indices = torch.stack([flattened_indices // f_subset.shape[1], flattened_indices % f_subset.shape[1]], dim=2) # (k, p, 2)

    # indices contain locations of tokens with highest feature activations in the original contexts
    indices = unflattened_indices.clone()
    indices[:, :, 1] = token_indices[unflattened_indices[:, :, 0], unflattened_indices[:, :, 1]]

    top_tokens_with_contexts = torch.stack([contexts[indices[:, :, 0], indices[:, :, 1]+i] for i in range(-context_on_each_side, context_on_each_side + 1)], dim=2)

    return top_values, top_tokens_with_contexts

def special_decode(tokens):
    """
    Decodes a list of tokens into a string. This function has the special feature of 
    highlighting the middle token in red, which is useful for identifying the token 
    with the highest feature activation.

    Input:
        tokens: List[str]. A list of tokens to decode.

    Output:
        str: The decoded string, with the middle token wrapped in red color.
    """
    assert len(tokens) % 2 == 1, "tokens must have odd length so that the middle token is unique"

    left_context = decode(tokens[:len(tokens)//2])
    mid_token = decode([tokens[len(tokens)//2]])
    right_context = decode(tokens[len(tokens)//2 + 1:])

    red_start = "\033[91m"
    color_end = "\033[0m"
    
    return f"{left_context}{red_start}{mid_token}{color_end}{right_context}"

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
    compile = False # TODO: Don't know why I needed to set compile to False before loading the model..
    # TODO: I dont know why the next 4 lines are needed. state_dict does not seem to have any keys with unwanted_prefix.
    unwanted_prefix = '_orig_mod.' 
    for key, val in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    gpt.load_state_dict(state_dict)
    gpt.eval()
    gpt.to(device)
    if compile:
        gpt = torch.compile(gpt) # requires PyTorch 2.0 (optional)
    config['block_size'] = block_size = gpt.config.block_size
    
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

    ## Get text data for evaluation 
    X, Y = get_text_batch(text_data, block_size=block_size, batch_size=eval_contexts) # (eval_contexts, block_size) 

    # Compute and store MLP activations on evaluation text data
    mlp_activations_storage = torch.tensor([], dtype=torch.float16)
    
    ## start evaluation
    num_eval_batches = eval_contexts // eval_batch_size
    for iter in range(num_eval_batches):    
        
        print(f'iter = {iter}/{num_eval_batches} in computation of mlp_acts for eval data')

        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = slice_fn(X).pin_memory().to(device, non_blocking=True) if device_type == 'cuda' else slice_fn(X).to(device)
        y = slice_fn(Y).pin_memory().to(device, non_blocking=True) if device_type == 'cuda' else slice_fn(Y).to(device)

        _, mlp_activations, batch_loss, batch_ablated_loss = gpt.forward_with_and_without_mlp(x, y)    
        mlp_activations_storage = torch.cat([mlp_activations_storage, mlp_activations.to(dtype=torch.float16, device='cpu')])

    token_indices = torch.randint(length_context_on_each_side, block_size - length_context_on_each_side, (eval_contexts, tokens_per_eval_context)) # (eval_contexts, tokens_per_eval_context)
    # TODO: This assumes that the tokens chosen for studying top activations are not on the extreme ends of any text in the context window of gpt
    # Can the generality we lose this way significantly affect our analysis? Put another way, could there be some interesting behavior on the edges of the context window?

    top_dict = {i: [] for i in range(n_features)}

    # feature_activation_counts = torch.zeros(n_features, dtype=torch.float32) # initiate with zeros

    for iter in range(num_eval_batches):
        print(f'{iter}/{num_eval_batches} for evaluation')

        # select batch of mlp activations, residual stream and y 
        if device_type == 'cuda':
            batch_mlp_activations = slice_fn(mlp_activations_storage).pin_memory().to(device, non_blocking=True) 
        else:
            batch_mlp_activations = slice_fn(mlp_activations_storage).to(device) # (eval_batch_size, block_size, n_ffwd)

        with torch.no_grad():
            output = autoencoder(batch_mlp_activations)
        
        batch_f = output['f'].to('cpu') # (eval_batch_size, block_size, n_features)
        batch_token_indices = slice_fn(token_indices) # (eval_batch_size, tokens_per_eval_context)
        batch_contexts = slice_fn(X)

        # restrict batch_contexts and batch_f on the subset of tokens specified by batch_token_indices
        batch_contexts_subset = torch.gather(batch_contexts, 1, batch_token_indices) # (eval_batch_size, tokens_per_eval_context)
        batch_f_subset = torch.gather(batch_f, 1, batch_token_indices.unsqueeze(-1).expand(-1, -1, n_features)) # (eval_batch_size, tokens_per_eval_context, n_features)
        
        # compute top feature activation values and contexts in which they appear
        batch_top_values, batch_top_contexts = top_activations_and_tokens(batch_f_subset, batch_token_indices, batch_contexts, length_context_on_each_side, k)
        for feature in range(n_features):
            for top_index in range(k):
                top_dict[feature].append((batch_top_values[top_index, feature].item(), special_decode(batch_top_contexts[top_index, feature].tolist())))
            top_dict[feature] = sorted(top_dict[feature], reverse=True)[:k] # keep only the top k values overall

    # save the dictionary of top activations and tokens in the same directory and sub-directory as the autoencoder
    top_dict_path = os.path.join(autoencoder_dir, autoencoder_subdir, 'top_dict.pkl')
    with open(top_dict_path, 'wb') as fp:
        pickle.dump(top_dict, fp)
        print(f'top_dict saved successfully to file {top_dict_path}')


    ## publish to html [optional]
    if publish_html:
        from publish_to_html import print_dict, ansi_to_html
        feature_activation_counts = autoencoder_ckpt['feature_activation_counts']
        eval_tokens = autoencoder_ckpt['config']['eval_tokens']
        high_density_neuron_indices = torch.arange(len(feature_activation_counts))[feature_activation_counts > modes_density_cutoff * eval_tokens]
        high_density_neurons = {key:val for key, val in top_dict.items() if key in high_density_neuron_indices}
        print(f'Number of high density neurons: {len(high_density_neurons)}')

        ultra_low_density_indices = torch.arange(len(feature_activation_counts))[(feature_activation_counts != 0) & (feature_activation_counts < modes_density_cutoff * eval_tokens)]
        ultra_low_density_neurons = {key:val for key, val in top_dict.items() if key in ultra_low_density_indices}
        print(f'Number of ultra-low density neurons: {len(ultra_low_density_neurons)}')

        
        for current_dict in [ultra_low_density_neurons, high_density_neurons]:
            for sort_by_activations in [True, False]:
                if sort_by_activations:
                    file_name = 'sorted_high_density_neurons' if current_dict == high_density_neurons else 'sorted_ultra_low_density_neurons'
                else: 
                    file_name = 'high_density_neurons' if current_dict == high_density_neurons else 'ultra_low_density_neurons'

                # TODO: provide manual_descriptions through a .json file?
                # manual_descriptions = update_manual_descriptions(high_density_neurons, manual_descriptions)

                # Write the content to an HTML file
                with open(os.path.join(autoencoder_dir, autoencoder_subdir, f'{file_name}.html'), 'w') as file:
                    original_stdout = sys.stdout 
                    # Capture the output of special_print directly into a string
                    output_stream = io.StringIO()
                    sys.stdout = output_stream  # Redirect stdout to the StringIO object
                    print_dict(current_dict, sort=sort_by_activations, neuron_descriptions=None)  # Call the function
                    sys.stdout = original_stdout  # Reset stdout to its original value

                    # The captured output is now in output_stream
                    captured_output = output_stream.getvalue()
                    html_content = f"""<!DOCTYPE html>
                    <html>
                    <head>
                        <title>{file_name}</title>
                    </head>
                    <body>
                        <p>{ansi_to_html(captured_output)}</p>
                    </body>
                    </html>"""
                    file.write(html_content)