
"""
Import a dictionary of feature activations from autoencoder_dir/autoencoder_subdir/feature_infos.pkl and write HTML pages 
A couple of sample runs:
python write_html.py --k=5 --num_intervals=3 --interval_exs=2 --dataset=shakespeare_char --autoencoder_subdir=1704914564.90-autoencoder-shakespeare_char
python write_html.py --k=5 --num_intervals=6 --interval_exs=3 --autoencoder_subdir=1705203324.45-autoencoder-openwebtext 
"""

import os
import tiktoken # needed to decode contexts to text
import random
import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict

def write_main_page(n_features):

    main = """
    <!DOCTYPE html>
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
        <h1>Feature Browser</h1>
        <p>Slide to select a neuron number (0 to 1023) or enter it below:</p>
        
        <!-- Slider Input -->
        <input type="range" id="pageSlider" min="0" max="1023" value="0" oninput="updateInputBox(this.value)">
        <span id="sliderValue">0</span>

        <!-- Input Box and Go Button -->
        <input type="number" id="pageNumberInput" min="0" max="1023" value="0">
        <button onclick="goToPage()">Go</button>

        <!-- Display Area for Page Content -->
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
                var pageNumber = parseInt(document.getElementById("pageNumberInput").value);
                if (pageNumber >= 0 && pageNumber <= 1023) {
                    document.getElementById("pageSlider").value = pageNumber;
                    updateInputBox(pageNumber);
                } else {
                    alert("Please enter a valid page number between 0 and 1023.");
                }
            }
        
            function loadPageContent(pageNumber) {
                var contentDiv = document.getElementById("pageContent");
        
                fetch('feature_pages/' + pageNumber + '.html')
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Page not found');
                        }
                        return response.text();
                    })
                    .then(data => {
                        var newData = data.replace(/src="(.+?)"/g, 'src="feature_pages/$1"');
                        contentDiv.innerHTML = newData;
                        // Save the current page number to localStorage
                        localStorage.setItem('currentPage', pageNumber);
                    })
                    .catch(error => {
                        contentDiv.innerHTML = '<p>Error loading page content.</p>';
                    });
            }
        
            window.addEventListener('load', () => {
                // Try to load the page number from the URL parameter if available
                const urlParams = new URLSearchParams(window.location.search);
                const pageFromURL = urlParams.get('page');
        
                // Determine the page to load: URL parameter, or localStorage, or default to 0
                let pageToLoad = pageFromURL !== null && !isNaN(parseInt(pageFromURL)) ? parseInt(pageFromURL) : parseInt(localStorage.getItem('currentPage') || 0);
                pageToLoad = Math.max(0, Math.min(1023, pageToLoad)); // Validate the page number
                
                document.getElementById("pageSlider").value = pageToLoad;
                document.getElementById("pageNumberInput").value = pageToLoad;
                document.getElementById("sliderValue").textContent = pageToLoad;
                loadPageContent(pageToLoad);
            });
        
            // Listen for keydown events on the whole document
            document.addEventListener('keydown', function(event) {
                const key = event.key;
                if (key === "ArrowLeft" || key === "ArrowRight") {
                    let currentPageNumber = parseInt(document.getElementById("pageNumberInput").value);
                    if (key === "ArrowLeft") {
                        // Decrement the page number, ensuring it doesn't go below 0
                        currentPageNumber = Math.max(0, currentPageNumber - 1);
                    } else if (key === "ArrowRight") {
                        // Increment the page number, ensuring it doesn't go above 1023
                        currentPageNumber = Math.min(1023, currentPageNumber + 1);
                    }
                    document.getElementById("pageSlider").value = currentPageNumber;
                    updateInputBox(currentPageNumber);
                }
            });
        </script>
        
    </body>
    </html>
    """
    return main

def write_tooltip_css_file():
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
            width: 280px; /* Increased width */
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 10px; /* Add horizontal padding if needed */
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -140px; /* Adjusted to half of the new width to keep it centered */
            opacity: 0;
            transition: opacity 0.3s;
            white-space: pre-wrap;
            overflow: hidden; /* Ensures the content does not spill outside the tooltip */
        }

        /* Show the tooltip content when hovering over the tooltip */
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }

        /* Style for the tooltip trigger text with the default color */
        .tooltip > span.default-color {
            background-color: #FFCC99; /* Light orange background color */
            color: #333; /* Dark text color for contrast */
            padding: 2px;
            border-radius: 4px;
        }

        /* Style for the tooltip trigger text with white color */
        .tooltip > span.white-color {
            background-color: #FFFFFF; /* White background color */
            color: #333; /* Dark text color for contrast */
            padding: 2px;
            border-radius: 4px;
        }
        """
        
    return tooltip_css

def create_main_html_page(n_features, dirpath=None):
    # create a directory to store feature information
    os.makedirs(os.path.join(dirpath, 'feature_pages'), exist_ok=True)
    # create a directory to store histograms of feature activations
    os.makedirs(os.path.join(dirpath, 'histograms'), exist_ok=True)
    # write a helper css file tooltip.css in autoencoder_subdir
    with open(os.path.join(dirpath, f'tooltip.css'), 'w') as file:
        file.write(write_tooltip_css_file()) 
    # write the main page for html
    with open(os.path.join(dirpath, 'main.html'), 'w') as file:
        file.write(write_main_page(n_features))
    print(f'created main page for HTML interface in {dirpath}')

def make_histogram(activations, density, feature_id, dirpath=None):
    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().numpy()
    plt.hist(activations, bins='auto')  # You can adjust the number of bins as needed
    plt.title(f'Activations (Density = {density:.4f}%)')
    plt.xlabel('Activation')
    plt.ylabel('Frequency')

    # Save the histogram as an image
    plt.savefig(os.path.join(dirpath, 'histograms', f'{feature_id}.png'))
    plt.close()

def feature_page_header(feature_id):
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
            </style>
    </head>
    <body>
    <br><br>
    <span style="color:blue;">
        <h2>Neuron # {feature_id}</h2>
    </span>
    """
    return header

def write_dead_feature_page(feature_id, dirpath=None):
    html_content = []
    # add page_header to list of texts
    html_content.append(feature_page_header(feature_id)) 
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
    for j in range(W):
        char = decode([tokens[j].item()])
        activation = activations[j].item()

        # for html rendering, replace newline character with its HTML counterpart
        char = char.replace('\n', '<span style="font-weight: normal;">&#x23CE;</span>')
        
        text_color = "default-color" if activation > 0 else "white-color"
        # TODO: Instead of writing with weird indentation, I should 
        # write with indentation first then remove indentation when appending
        # text to html_content. This is purely so that the s
        single_token_text = f"""
        <div class="tooltip"> 
            <span class="{text_color}"> {char.replace(' ', '&nbsp;')} </span> 
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
    html_content.append(feature_page_header(feature_id)) 

    # add histogram of feature activations
    if os.path.exists(os.path.join(dirpath, 'histograms', f'{feature_id}.png')):
        html_content.append(f"""<img src=\"../histograms/{feature_id}.png\" alt=\"Feature Activations Histogram\">
        <br>""")

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
    html_content.append(feature_page_header(feature_id)) 

    html_content.append("""
    <h2> <span style="color:blue;">  Ultralow Density Neuron </span> </h2> 
    """)

    # add histogram of feature activations
    if os.path.exists(os.path.join(dirpath, 'histograms', f'{feature_id}.png')):
        html_content.append(f"""<img src=\"../histograms/{feature_id}.png\" alt=\"Feature Activations Histogram\">
        <br>""")

    # include a section on top activations
    html_content.append("""
    <h3>  Top Activations </h3> 
    """)
    html_content.append(write_activations_section(decode, top_acts_data))

    # include the end of the HTML page
    html_content.append("</body> </html>")
    with open(os.path.join(dirpath, 'feature_pages', f'{feature_id}.html'), 'w') as file:
        file.write("".join(html_content))