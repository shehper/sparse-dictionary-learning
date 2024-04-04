import os

def write_main_page(n_features):

    main = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title> Feature Visualization </title>
        <style>
            body {{
                text-align: center; /* Center content */
                font-family: Arial, sans-serif; /* Font style */
            }}
            #pageSlider, #pageNumberInput, button {{
                margin-top: 20px; /* Space above elements */
                margin-bottom: 20px; /* Space below elements */
                width: 50%; /* Width of the slider and input box */
                max-width: 400px; /* Maximum width */
            }}
            #pageNumberInput, button {{
                width: auto; /* Auto width for input and button */
                padding: 5px 10px; /* Padding inside input box and button */
                font-size: 16px; /* Font size */
            }}
            #pageContent {{
                margin-top: 20px; /* Space above page content */
                width: 80%; /* Width of the content area */
                margin-left: auto; /* Center the content area */
                margin-right: auto; /* Center the content area */
            }}
        </style>
    </head>
    <body>
        <h1>Feature Browser</h1>
        <p>Slide to select a neuron number (0 to {n_features-1}) or enter it below:</p>
        
        <!-- Slider Input -->
        <input type="range" id="pageSlider" min="1" max="{n_features-1}" value="0" oninput="updateInputBox(this.value)">
        <span id="sliderValue">0</span>

        <!-- Input Box and Go Button -->
        <input type="number" id="pageNumberInput" min="0" max="{n_features-1}" value="0">
        <button onclick="goToPage()">Go</button>

        <!-- Display Area for Page Content -->
        <div id="pageContent">
            <!-- Content will be loaded here -->
        </div>

        <script>
            function updateInputBox(value) {{
                document.getElementById("sliderValue").textContent = value;
                document.getElementById("pageNumberInput").value = value;
                loadPageContent(value);
            }}
        
            function goToPage() {{
                var pageNumber = parseInt(document.getElementById("pageNumberInput").value);
                if (pageNumber >= 0 && pageNumber <= {n_features-1}) {{
                    document.getElementById("pageSlider").value = pageNumber;
                    updateInputBox(pageNumber);
                }} else {{
                    alert("Please enter a valid page number between 0 and {n_features-1}.");
                }}
            }}
        
            function loadPageContent(pageNumber) {{
                var contentDiv = document.getElementById("pageContent");
        
                fetch('feature_pages/' + pageNumber + '.html')
                    .then(response => {{
                        if (!response.ok) {{
                            throw new Error('Page not found');
                        }}
                        return response.text();
                    }})
                    .then(data => {{
                        var newData = data.replace(/src="(.+?)"/g, 'src="feature_pages/$1"');
                        contentDiv.innerHTML = newData;
                        // Save the current page number to localStorage
                        localStorage.setItem('currentPage', pageNumber);
                    }})
                    .catch(error => {{
                        contentDiv.innerHTML = '<p>Error loading page content.</p>';
                    }});
            }}
        
            window.addEventListener('load', () => {{
                // Try to load the page number from the URL parameter if available
                const urlParams = new URLSearchParams(window.location.search);
                const pageFromURL = urlParams.get('page');
        
                // Determine the page to load: URL parameter, or localStorage, or default to 0
                let pageToLoad = pageFromURL !== null && !isNaN(parseInt(pageFromURL)) ? parseInt(pageFromURL) : parseInt(localStorage.getItem('currentPage') || 0);
                pageToLoad = Math.max(0, Math.min({n_features-1}, pageToLoad)); // Validate the page number
                
                document.getElementById("pageSlider").value = pageToLoad;
                document.getElementById("pageNumberInput").value = pageToLoad;
                document.getElementById("sliderValue").textContent = pageToLoad;
                loadPageContent(pageToLoad);
            }});
        
            // Listen for keydown events on the whole document
            document.addEventListener('keydown', function(event) {{
                const key = event.key;
                if (key === "ArrowLeft" || key === "ArrowRight") {{
                    let currentPageNumber = parseInt(document.getElementById("pageNumberInput").value);
                    if (key === "ArrowLeft") {{
                        // Decrement the page number, ensuring it doesn't go below 0
                        currentPageNumber = Math.max(0, currentPageNumber - 1);
                    }} else if (key === "ArrowRight") {{
                        // Increment the page number, ensuring it doesn't go above {n_features-1}
                        currentPageNumber = Math.min({n_features-1}, currentPageNumber + 1);
                    }}
                    document.getElementById("pageSlider").value = currentPageNumber;
                    updateInputBox(currentPageNumber);
                }}
            }});
        </script>
        
    </body>
    </html>
    """
    return main

def write_tooltip_css_file():
    tooltip_css = f"""/* Style for the tooltip */
        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: pointer;
            margin-right: -4px;
        }}

        /* Style for the tooltip trigger text */
        .tooltip > span {{
            background-color: #FFCC99; /* Light orange background color */
            color: #333; /* Dark text color for contrast */
            padding: 0px; /* Add padding to make the background more prominent */
            border-radius: 0px; /* Optional: Adds rounded corners to the background */
        }}

        /* Style for the tooltip content */
        .tooltip .tooltiptext {{
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
        }}

        /* Show the tooltip content when hovering over the tooltip */
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}

        /* Style for the tooltip trigger text with the default color */
        .tooltip > span.default-color {{
            background-color: #FFCC99; /* Light orange background color */
            color: #333; /* Dark text color for contrast */
            padding: 0px;
            border-radius: 0px;
        }}

        /* Style for the tooltip trigger text with white color */
        .tooltip > span.white-color {{
            background-color: #FFFFFF; /* White background color */
            color: #333; /* Dark text color for contrast */
            padding: 0px;
            border-radius: 0px;
        }}
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