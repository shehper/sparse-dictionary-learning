
"""
Import a dictionary of feature activations from autoencoder_dir/autoencoder_subdir/feature_infos.pkl and write HTML pages 
A couple of sample runs:
python write_html.py --k=5 --num_intervals=3 --interval_exs=2 --dataset=shakespeare_char --autoencoder_subdir=1704914564.90-autoencoder-shakespeare_char
python write_html.py --k=5 --num_intervals=6 --interval_exs=3 --autoencoder_subdir=1705203324.45-autoencoder-openwebtext 
"""

import os
import numpy as np
import pickle # needed to load meta.pkl
import tiktoken # needed to decode contexts to text
import random
import matplotlib.pyplot as plt

# hyperparameters --- same as train_sae.py except a few maybe # TODO: do I need all of them?
device = 'cpu' # change it to cpu
seed = 1442
dataset = 'openwebtext' 
autoencoder_dir = 'out_autoencoder' # directory containing weights of various trained autoencoder models
autoencoder_subdir = '' # subdirectory containing the specific model to consider
context_length = 10 # number of tokens to print/save on either side of the token with feature activation. 
k = 15 # number of top activations
num_intervals = 11 # number of intervals to divide activations in; = 11 in Anthropic's work
interval_exs = 5 # number of examples to sample from each interval of activations 
make_histogram = False

## define a function that converts tokens to html text
def context_to_html(text, max_act_this_text):
    out = """"""
    for token, act in text: 
        token = decode([token]).replace('\n', '<span style="font-weight: normal;">&#x23CE;</span>').replace(' ', '&nbsp;')
        if act == 0:
            out += token
        elif act == max_act_this_text: # if the 
            out +=  f"""<div class="tooltip"> <span><strong>{token}</strong></span>
                            <span class="tooltiptext"> Activation: {act:.4f} </span>
                        </div>"""
        else: 
            out +=  f"""<div class="tooltip"> <span> {token} </span> 
                            <span class="tooltiptext"> Activation: {act:.4f} </span>
                        </div>"""
    out += "<br>"
    return out 

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
                                // Update relative paths for images and links to be correct
                                // Assuming 'pages/' is the correct path from main.html to the images
                                var newData = data.replace(/src="(.+?)"/g, 'src="pages/$1"');
                                contentDiv.innerHTML = newData;
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
    return main

def tooltip_css():
    tooltip_css = """/* Style for the tooltip */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }

        /* Style for the tooltip trigger text */
        .tooltip > span {
            background-color: #FFCC99; /* Light orange background color */
            color: #333; /* Dark text color for contrast */
            padding: 2px; /* Add padding to make the background more prominent */
            border-radius: 4px; /* Optional: Adds rounded corners to the background */
        }

        /* Style for the tooltip content */
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 140px;
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
            white-space: pre; /* Preserve whitespace */
        }

        /* Show the tooltip content when hovering over the tooltip */
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        """
        
    return tooltip_css

def feature_page(feature_info):

    # get info for the current feature
    curr_info = feature_info # list of tuples (float, list) where the latter list is of acts and tokens
    curr_info.sort(reverse=True) # sort by maximum activation value in descending order

    # start html text for this neuron 
    html_ = """<!DOCTYPE html> <html lang="en"> <head> <meta charset="UTF-8"> 
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link rel="stylesheet" href="tooltip.css"> </head> <body> <br>  """
    html_ += f"""<span style="color:blue;"> <h2>  Neuron # {i} </h2> </span>"""
    if len(curr_info) == 0: # the neuron is dead if the list is empty
        html_ += f"""<span style="color:red;"> <h2>  This neuron is dead. </h2> </span> """
        return html_

    max_act = curr_info[0][0] # maximum activation value for this feature
    html_ += f"""<h3> TOP ACTIVATIONS, MAX ACTIVATION: {max_act:.4f} </h3> """
    for max_act_this_text, text in curr_info[:k]: # iterate over top k examples
        # text is a list of tuples (token, activation value of the token)
        # top_act_this_text is the maximum of all activation values in this text
        # convert the context and token into an HTML text
        html_ += context_to_html(text, max_act_this_text)

    # if there are enough examples to create subsample intervals, create them
    if len(curr_info) > num_intervals * interval_exs:
        id, start = 1, 0 # id tracks the subsample interval number, start is the index of the start of current interval in curr_info
        for j, (max_act_this_text, _) in enumerate(curr_info):
            if id < num_intervals and max_act_this_text < max_act * (num_intervals - i) / num_intervals: # have 
                end = j # end is the index of the end of current interval in curr_info
                exs = random.choices(curr_info[start: end], k=interval_exs) # pick interval_exs random examples from this subsample
                exs.sort(reverse=True) 
                html_ += f"""<h3> SUBSBAMPLE INTERVAL {id-1}, MAX ACTIVATION: {exs[0][0]:.4f} </h3> """
                for max_act_this_text, text in exs: # write to html
                    html_ += context_to_html(text, max_act_this_text)            
                start = end
                id += 1

    ## Plot a histogram of activations
    histogram_path = os.path.join(autoencoder_dir, autoencoder_subdir, 'pages', f'histogram_{i}.png')
    if make_histogram:
        all_acts = [] # all non-zero activation values for this feature
        for (_, text) in curr_info: # iterate through all examples and collect activation values
            for _, act in text:
                if act != 0:
                    all_acts += [act]
        plt.hist(all_acts, bins='auto')  # You can adjust the number of bins as needed
        plt.title('Feature Activations Histogram')
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency')

        ## Save the histogram as an image
        plt.savefig(histogram_path)
        plt.close()

    # if a histogram file exists, add it to the HTML page
    if os.path.exists(histogram_path):
        html_ += f"<img src=\"{os.path.basename(histogram_path)}\" alt=\"Feature Activations Histogram\">" 

    html_ += """</body> </html>"""
    return html_

if __name__ == '__main__':

    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    ## load feature_info.pkl
    print(f'loading feature_info.pkl...')
    with open(os.path.join(autoencoder_dir, autoencoder_subdir, 'feature_info.pkl'), 'rb') as f:
        feature_infos = pickle.load(f)
    n_features = len(feature_infos)
    print(f'load successful: feature_info.pkl from {os.path.join(autoencoder_dir, autoencoder_subdir)}')

    ## load tokenizer used to train the gpt model
    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    current_dir = os.path.abspath('.')
    meta_path = os.path.join(os.path.dirname(current_dir), 'transformer', 'data', dataset, 'meta.pkl')
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

    # create a directory to store pages
    os.makedirs(os.path.join(autoencoder_dir, autoencoder_subdir, 'pages'), exist_ok=True)
    # write a helper css file tooltip.css in autoencoder_subdir
    with open(os.path.join(autoencoder_dir, autoencoder_subdir, f'tooltip.css'), 'w') as file:
        file.write(tooltip_css()) 
    # write the main page for html
    with open(os.path.join(autoencoder_dir, autoencoder_subdir, 'main.html'), 'w') as file:
        file.write(main_page(n_features))
    print(f'wrote tooltip.css and main.html in {os.path.join(autoencoder_dir, autoencoder_subdir)}')

    # write an html page for each feature
    for i, feature_info in enumerate(feature_infos):
        if i % 100 == 0:
            print(f'working on neurons {i} through {i+99}')
        with open(os.path.join(autoencoder_dir, autoencoder_subdir, 'pages', f'page{i}.html'), 'w') as file:
            file.write(feature_page(feature_infos[i])) 
