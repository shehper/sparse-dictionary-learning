
"""
Import a dictionary of feature activations from autoencoder_dir/autoencoder_subdir/feature_infos.pkl and write HTML pages 
A couple of sample runs:
python write_html.py --k=5 --num_intervals=3 --interval_exs=2 --dataset=shakespeare_char --autoencoder_subdir=1704914564.90-autoencoder-shakespeare_char
python write_html.py --k=5 --num_intervals=6 --interval_exs=3 --autoencoder_subdir=1705203324.45-autoencoder-openwebtext 
"""

import os
import torch
from tensordict import TensorDict

def write_feature_page_header(feature_id):
    header = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="tooltip.css">
        <style>
        body {{
            text-align: center; /* Center content */
        }}
        .content-container {{
            display: flex;
            justify-content: center;
            align-items: flex-start; /* Adjust this as needed */
            margin-top: 20px; /* Adjust space from the top */
        }}
        .image-container {{
            flex: 1; /* Adjust as needed */
            text-align: center; /* Center image */
        }}
        .text-container {{
            flex: 1; /* Adjust as needed */
            text-align: left; /* Align text to the left */
            padding-left: 20px; /* Space between image and text */
        }}
    </style>
    </head>
    <body>
    <br><br>
    """
    return header

def write_dead_feature_page(feature_id, dirpath=None):
    html_content = []
    # add page_header to list of texts
    html_content.append(write_feature_page_header(feature_id)) 
    # add dead neuron text
    html_content.append("""<span style="color:red;"> 
                            <h2>  Dead Neuron. </h2> 
                            </span> 
                            </body> 
                            </html>""")
    # write the HTML file
    with open(os.path.join(dirpath, 'feature_pages', f'{feature_id}.html'), 'w') as file:
            file.write("".join(html_content))

def write_activation_example(decode, tokens, activations):
    assert isinstance(tokens, torch.Tensor) and isinstance(activations, torch.Tensor), "expect inputs to be torch tensors"
    assert tokens.ndim == 1 and activations.ndim == 1, "expect tokens and acts to be 1d tensors"
    html_content = []
    W = tokens.shape[0]
    mid_token_index = (W-1)//2

    start_bold_text = lambda j, mid_token_index: "<b>" if j == mid_token_index else ""
    end_bold_text = lambda j, mid_token_index: "</b>" if j == mid_token_index else ""

    for j in range(W):
        char = decode([tokens[j].item()])
        activation = activations[j].item()

        # for html rendering, replace newline character with its HTML counterpart
        char = char.replace('\n', '<span style="font-weight: normal;">&#x23CE;</span>')
        
        text_color = "default-color" if activation > 0 else "white-color"
        single_token_text = f"""
        <div class="tooltip"> 
            <span class="{text_color}"> {start_bold_text(j, mid_token_index)} {char.replace(' ', '&nbsp;')} {end_bold_text(j, mid_token_index)} </span> 
            <span class="tooltiptext"> 
                Token: {char}
                Activation: {activation:.4f} 
            </span>
        </div>"""
        html_content.append(single_token_text)
    html_content.append("""<br>""")
    return "".join(html_content)


def write_activations_section(decode, examples_data):
    assert examples_data.ndim == 2, "input must be two dimensional, shape: (X, W) or (k, W)"
    n, W = examples_data.shape

    html_content = []
    html_content.append(f"""
    <h3>  Max Activation = {examples_data[0]['feature_acts'][(W-1)//2]:.4f} </h3>
    """)

    for i in range(n):
        html_content.append(write_activation_example(decode, 
                                                    tokens=examples_data["tokens"][i],
                                                    activations=examples_data["feature_acts"][i]))
    return "".join(html_content)
        
        
def write_alive_feature_page(feature_id, decode, top_acts_data, sampled_acts_data, dirpath=None):

    print(f'writing feature page for feature # {feature_id}')
    
    assert isinstance(top_acts_data, TensorDict), "expect top activation data to be presented in a TensorDict" 
    assert top_acts_data.ndim == 2, "expect top activation data TensorDict to be 2-dimensional, shape: (k, W)"

    assert isinstance(sampled_acts_data, TensorDict), "expect samples activation data to be presented in a TensorDict" 
    assert sampled_acts_data.ndim == 3, "expect sampled activation data TensorDict to be 3-dimensional, shape: (I, X, W)"

    assert 'tokens' in top_acts_data.keys() and 'feature_acts' in top_acts_data.keys() and \
        'tokens' in sampled_acts_data.keys() and 'feature_acts' in sampled_acts_data.keys(), \
        "expect input TensorDicts to have tokens and features_acts keys"
    
    html_content = []

    # add page_header to the HTML page
    html_content.append(write_feature_page_header(feature_id)) 

    # add histogram of feature activations
    if os.path.exists(os.path.join(dirpath, 'histograms', f'{feature_id}.png')):
        html_content.append(f"""<div class="content-container">
        <div class="image-container">
            <img src="../histograms/{feature_id}.png" alt="Feature Activations Histogram">
        </div>""")

    # add feature #, and the information that it is an ultralow density neuron
    html_content.append(f"""<div class="text-container">
        <h2 style="color:blue;">Neuron # {feature_id}</h2> """)

    # include a section on top activations
    html_content.append("""
    <h3>  Top Activations </h3> 
                        """)
    html_content.append(write_activations_section(decode, top_acts_data))

    # include a section on sampled activations
    I = sampled_acts_data.shape[0] # number of intervals
    for i in range(I):
        if i < I - 1:
            html_content.append(f"<h3>  Subsample Interval {i} </h3> ")
        else:
            html_content.append(f"<h3> Bottom Activations </h3>")
        html_content.append(write_activations_section(decode, sampled_acts_data[i]))

    # include the end of the HTML page
    html_content.append("</body> </html>")
    with open(os.path.join(dirpath, 'feature_pages', f'{feature_id}.html'), 'w') as file:
        file.write("".join(html_content))

def write_ultralow_density_feature_page(feature_id, decode, top_acts_data, dirpath=None):

    print(f'writing feature page for feature # {feature_id}')
    
    assert isinstance(top_acts_data, TensorDict), "expect top activation data to be presented in a TensorDict" 
    assert top_acts_data.ndim == 2, "expect top activation data TensorDict to be 2-dimensional, shape: (n, W)"

    assert 'tokens' in top_acts_data.keys() and 'feature_acts' in top_acts_data.keys() and \
        "expect input TensorDict to have tokens and features_acts keys"
    
    html_content = []

    # add page_header to the HTML page
    html_content.append(write_feature_page_header(feature_id)) 

    # add histogram of feature activations
    if os.path.exists(os.path.join(dirpath, 'histograms', f'{feature_id}.png')):
        html_content.append(f"""<div class="content-container">
        <div class="image-container">
            <img src="../histograms/{feature_id}.png" alt="Feature Activations Histogram">
        </div>""")

    # add feature #, and the information that it is an ultralow density neuron
    html_content.append(f"""<div class="text-container">
        <h2 style="color:blue;">Neuron # {feature_id}</h2>
        <h2> <span style="color:blue;">Ultralow Density Neuron</span> </h2> """)


    # include a section on top activations
    html_content.append("""
    <h3>  Top Activations </h3> 
    """)
    html_content.append(write_activations_section(decode, top_acts_data))

    # include the end of the HTML page
    html_content.append("</body> </html>")
    with open(os.path.join(dirpath, 'feature_pages', f'{feature_id}.html'), 'w') as file:
        file.write("".join(html_content))