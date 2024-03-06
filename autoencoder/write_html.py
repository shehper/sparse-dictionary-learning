
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

def write_main_page(n_features):

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
                <h1>Feature Browser</h1>
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

                        fetch('feature_pages/' + pageNumber + '.html')
                            .then(response => {
                                if (!response.ok) {
                                    throw new Error('Page not found');
                                }
                                return response.text();
                            })
                            .then(data => {
                                // Update relative paths for images and links to be correct
                                // Assuming 'feature_pages/' is the correct path from main.html to the images
                                var newData = data.replace(/src="(.+?)"/g, 'src="feature_pages/$1"');
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
    header = f"""<!DOCTYPE html> <html lang="en"> <head> <meta charset="UTF-8"> 
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link rel="stylesheet" href="tooltip.css"> </head> <body> <br> <br> 
                <span style="color:blue;"> <h2>  Neuron # {feature_id} </h2> </span>"""
    return header

def write_dead_feature_page(feature_id, dirpath=None):
    page_header = feature_page_header(feature_id)
    page = page_header + f"""<span style="color:red;"> <h2>  Dead Neuron. </h2> </span> """
    with open(os.path.join(dirpath, 'feature_pages', f'{feature_id}.html'), 'w') as file:
            file.write(page)

def write_ultralow_density_feature_page(feature_id, dirpath=None):
    page_header = feature_page_header(feature_id)
    page = page_header + f"""<span style="color:red;"> <h2>  Ultralow Density Neuron. </h2> </span> """
    with open(os.path.join(dirpath, 'feature_pages', f'{feature_id}.html'), 'w') as file:
            file.write(page)

## define a function that converts tokens to html text
def convert_text_to_html(text, max_act_this_text, decode):
    out = """"""
    for token, act in text: 
        token = decode([token.item()]).replace('\n', '<span style="font-weight: normal;">&#x23CE;</span>').replace(' ', '&nbsp;')
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


def write_alive_feature_page(feature_id, dirpath=None):
    page_header = feature_page_header(feature_id)
    page = page_header + """The rest of the information goes here."""
     # if a histogram file exists, add it to the HTML page
    histogram_path = os.path.join(dirpath, 'histograms', f'{feature_id}.png')
    if os.path.exists(histogram_path):
        page = page + f"<img src=\"../histograms/{os.path.basename(histogram_path)}\" alt=\"Feature Activations Histogram\">" 
    page = page + """</body> </html>"""
    with open(os.path.join(dirpath, 'feature_pages', f'{feature_id}.html'), 'w') as file:
            file.write(page)

def write_feature_page(feature_id, feature_info, decode, make_histogram, k, num_intervals, interval_exs, autoencoder_dir, autoencoder_subdir):


    max_act = curr_info[0][0] # maximum activation value for this feature
    html_ += f"""<h3> TOP ACTIVATIONS, MAX ACTIVATION: {max_act:.4f} </h3> """
    for max_act_this_text, text in curr_info[:k]: # iterate over top k examples
        # text is a list of tuples (token, activation value of the token)
        # top_act_this_text is the maximum of all activation values in this text
        # convert the context and token into an HTML text
        html_ += convert_text_to_html(text, max_act_this_text, decode)

    # if there are enough examples to create subsample intervals, create them
    if len(curr_info) > num_intervals * interval_exs:
        id, start = 1, 0 # id tracks the subsample interval number, start is the index of the start of current interval in curr_info
        for j, (max_act_this_text, _) in enumerate(curr_info):
            if id < num_intervals and max_act_this_text < max_act * (num_intervals - feature_id) / num_intervals: # have 
                end = j # end is the index of the end of current interval in curr_info
                exs = random.choices(curr_info[start: end], k=interval_exs) # pick interval_exs random examples from this subsample
                exs.sort(reverse=True) 
                html_ += f"""<h3> SUBSBAMPLE INTERVAL {id-1}, MAX ACTIVATION: {exs[0][0]:.4f} </h3> """
                for max_act_this_text, text in exs: # write to html
                    html_ += convert_text_to_html(text, max_act_this_text, decode)            
                start = end
                id += 1
        

    # if a histogram file exists, add it to the HTML page
    if os.path.exists(os.path.join(autoencoder_dir, autoencoder_subdir, 'histograms', f'histogram_{feature_id}.png')):
        html_ += f"<img src=\"{os.path.basename(histogram_path)}\" alt=\"Feature Activations Histogram\">" 

    html_ += """</body> </html>"""
    return html_