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
import random
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
modes_density_cutoff = 1e-3
publish_html = False

slice_fn = lambda storage: storage[iter * eval_batch_size: (iter + 1) * eval_batch_size]

def main_page(n_features):
    
    main = """<!DOCTYPE html>
            <html>
            <head>
                <title> Feature Visualization </title>
                <style>
                    body {
                        text-align: center; /* Center content */
                        font-family: Arial, sans-serif; /* Font style */
                    }
                    #pageSlider, #pageNumberInput, button {
                        margin-top: 20px; /* Space above elements */
                        margin-bottom: 20px; /* Space below elements */
                        width: 50%; /* Width of the slider and input box */
                        max-width: 400px; /* Maximum width */
                    }
                    #pageNumberInput, button {
                        width: auto; /* Auto width for input and button */
                        padding: 5px 10px; /* Padding inside input box and button */
                        font-size: 16px; /* Font size */
                    }
                    #pageContent {
                        margin-top: 20px; /* Space above page content */
                        width: 80%; /* Width of the content area */
                        margin-left: auto; /* Center the content area */
                        margin-right: auto; /* Center the content area */
                    }
                </style>
            </head>
            <body>
                <h1>Feature browser</h1>
                <p>Slide to select a neuron number""" + f""" (0 to {n_features-1}) or enter it below:</p>
                
                <!-- Slider Input -->
                <input type="range" id="pageSlider" min="0" max="{n_features-1}" value="0" oninput="updateInputBox(this.value)">
                <span id="sliderValue">0</span>

                <!-- Input Box and Go Button -->
                <input type="number" id="pageNumberInput" min="0" max="{n_features-1}" value="0">
                <button onclick="goToPage()">Go</button>

                <!-- Display Area for Page Content""" + """ -->
                <div id="pageContent">
                    <!-- Content will be loaded here -->
                </div>

                <script>
                    function updateInputBox(value) {
                        document.getElementById("sliderValue").textContent = value;
                        document.getElementById("pageNumberInput").value = value;
                        loadPageContent(value);
                    }

                    function goToPage() {
                        var pageNumber = document.getElementById("pageNumberInput").value;
                        if (pageNumber >= 0 && pageNumber""" + f""" <= {n_features-1})""" + """ {
                            document.getElementById("pageSlider").value = pageNumber;
                            updateInputBox(pageNumber);
                        } else {
                            alert("Please enter a valid page number between""" + f""" 0 and {n_features-1}""" + """.");
                        }
                    }

                    function loadPageContent(pageNumber) {
                        var contentDiv = document.getElementById("pageContent");

                        fetch('pages/page' + pageNumber + '.html')
                            .then(response => {
                                if (!response.ok) {
                                    throw new Error('Page not found');
                                }
                                return response.text();
                            })
                            .then(data => {
                                contentDiv.innerHTML = data;
                            })
                            .catch(error => {
                                contentDiv.innerHTML = '<p>Error loading page content.</p>';
                            });
                    }

                    // Initial load of Page 0 content
                    loadPageContent(0);
                    
                </script>
            </body>
            </html>

            """
    
    with open(os.path.join(autoencoder_dir, autoencoder_subdir, 'main.html'), 'w') as file:
        file.write(main)

def tooltip_css():
    tooltip_css = """/* Style for the tooltip */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        
        /* Style for the tooltip content */
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 120px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        /* Show the tooltip content when hovering over the tooltip */
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        """
        
    with open(os.path.join(autoencoder_dir, autoencoder_subdir, f'tooltip.css'), 'w') as file:
        file.write(tooltip_css) 

page_header = f"""<!DOCTYPE html>
                <html lang="en">
                <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link rel="stylesheet" href="tooltip.css">
                </head>
                <body>
                <br> 
                """

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
    # Keep a dictionary feature_activations = {i: [] for i in range(n_features)]
    # Whenever a feature activation is non-zero, add it to the list as (feature_activation, token with context around it)
    # At the very end, sample 
    # TODO: I think one thing that I should also try to keep track of is the location of the start of each context in train.bin
    # Why? Well, If I intend to compute correlation between contexts of top activations, this might be needed. 
    # Otherwise, I could just compute it from my list feature_activations[i].
    # By the way, I should look at MMCS/cosine similarity in the paper. They might already be doing what I am intending to do. 

    feature_activations = {i: [] for i in range(n_features)}

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

        for i in range(n_features):
            curr_f = batch_f[:, :, i] # (eval_batch_size, block_size)
            # now pick 10 random tokens
            sample_idx = torch.randint(context_length, block_size - context_length, (eval_batch_size, tokens_per_eval_context)) # (eval_batch_size, tokens_per_eval_context)

            # evaluate curr_f on these tokens
            curr_f_subset = torch.gather(curr_f, 1, sample_idx) # (eval_batch_size, tokens_per_eval_context)
            for k in range(eval_batch_size):
                for m in range(tokens_per_eval_context):
                    if curr_f_subset[k, m] != 0:
                        sample_c_idx = [l for l in range(sample_idx[k, m] - context_length, sample_idx[k, m] + context_length + 1)]
                        context = batch_contexts[k, sample_c_idx]
                        f_acts = batch_f[k, sample_c_idx, i]
                        context_acts = [(s, t) for s, t in zip(context, f_acts)]
                        feature_activations[i] += [(curr_f_subset[k, m].item(), context_acts)]

    with open(os.path.join(autoencoder_dir, autoencoder_subdir, 'feature_info.pkl'), 'wb') as f:
        pickle.dump(feature_activations, f)

    if publish_html:
        os.makedirs(os.path.join(autoencoder_dir, autoencoder_subdir, 'pages'), exist_ok=True)
        main_page(n_features)
        tooltip_css()

        for i in range(n_features):
            print(f'writing html page for neuron number {i}')
            feature_activations[i].sort(reverse=True) # sort the list of activations in a descending order
            top_acts = feature_activations[i][:k] # get top k elements
                
            html_ = page_header
            html_ += f"""<span style="color:blue;"> <h2>  Neuron # {i} </h2> </span>"""
            if len(top_acts) == 0:
                html_ += f"""<span style="color:red;"> <h2>  This neuron is dead. </h2> </span> """
            else:
                html_ += f"""<h3> TOP ACTIVATIONS, MAX ACTIVATION: {top_acts[0][0]:.4f} </h3> """
                for feature_act, context_acts in top_acts:
                    for token, act in context_acts:
                        if act == 0:
                            html_ += f"{decode([token.item()])}"
                        else: 
                            html_ +=  f"""<div class="tooltip">
                                            <span style="color:red;"> {decode([token.item()])} </span>
                                            <span class="tooltiptext"> Activation: {act:.4f} </span>
                                        </div>"""
                    html_ += "<br>"

            curr_data = feature_activations[i] # sorted list of tuples (float, list) where the latter list is of acts and tokens
            
            if len(curr_data) > num_intervals * interval_exs: # we must have enough examples to sample from different intervals
                top_val = curr_data[0][0]
                splits = [i * top_val / num_intervals for i in range(num_intervals, 0, -1)] # get bounds 
                index = 1
                start, end = 0, 0
                for j, (act, _) in enumerate(curr_data):
                    if index < len(splits) and act < splits[index]:
                        end = j
                        out = random.choices(curr_data[start: end], k=interval_exs)
                        out.sort(reverse=True)
                        html_ += f"""<h3> SUBSBAMPLE INTERVAL {index-1}, MAX ACTIVATION: {out[0][0]:.4f} </h3> """
                        for feature_act, context_acts in out:
                            for token, act in context_acts:
                                if act == 0:
                                    html_ += f"{decode([token.item()])}"
                                else: 
                                    html_ +=  f"""<div class="tooltip">
                                                    <span style="color:red;"> {decode([token.item()])} </span>
                                                    <span class="tooltiptext"> Activation: {act:.4f} </span>
                                                </div>"""
                            html_ += "<br>"
                        
                        start = end
                        index += 1
            
            html_ += """</body>
                        </html>"""
            with open(os.path.join(autoencoder_dir, autoencoder_subdir, 'pages', f'page{i}.html'), 'w') as file:
                file.write(html_) 


        # TODO: plot histogram of feature activations

        

