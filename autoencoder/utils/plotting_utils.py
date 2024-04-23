"""
Three different histogram functions. The difference lies in whether to save the histogram image on disk or not,
color scheme and axes labels.
These can perhaps be combined into one function, but leaving it as it is for now.
"""
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import torch
import os

def make_density_histogram(data, bins='auto'):
    """Makes a histogram image from the provided data and returns it.
    We use it in train.py to plot feature density histograms and log them with W&B."""
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins)
    ax.set_title('Histogram')
    plt.tight_layout()

    buf = BytesIO()  # create a BytesIO buffer
    fig.savefig(buf, format='png')  # save the plot to the buffer in PNG format
    buf.seek(0)  # rewind the buffer to the beginning
    image = Image.open(buf)  # open the image from the buffer

    plt.close(fig)  # close the figure to free memory
    return image

def make_activations_histogram(activations, density, feature_id, dirpath=None):
    """makes a histogram of activations and saves it on the disk
    we later include the histogram in the feature browser"""
    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().numpy()
    plt.hist(activations, bins='auto')  # You can adjust the number of bins as needed
    plt.title(f'Activations (Density = {density:.4f}%)')
    plt.xlabel('Activation')
    plt.ylabel('Frequency')

    # Save the histogram as an image
    image_path = os.path.join(dirpath, 'activations_histograms', f'{feature_id}.png')
    plt.savefig(image_path)
    plt.close()

def make_logits_histogram(logits, feature_id, dirpath=None):
    """
    Makes a histogram of logits for a given feature and saves it as a PNG file
    Input: 
        logits: a torch tensor of shape (vocab_size,)
        feature_id: int 
        dirpath: histogram is saved as dirpath/logits_histograms/feature_id.png
    """
    plt.hist(logits.cpu().numpy(), bins='auto')  # You can adjust the number of bins as needed

    image_path = os.path.join(dirpath, 'logits_histograms', f'{feature_id}.png')
    plt.savefig(image_path)
    plt.close()