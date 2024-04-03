import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def make_histogram_image(data, bins='auto'):
    """Generates a histogram image from the provided data."""
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