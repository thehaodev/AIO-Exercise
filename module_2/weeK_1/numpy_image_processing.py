import gdown
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from gdown.exceptions import FileURLRetrievalError
import numpy as np


def lightness(img: np.array):
    result = img.copy()
    max_rgb = np.max(img, axis=-1, keepdims=True)
    min_rgb = np.min(img, axis=-1, keepdims=True)
    result[:] = (max_rgb + min_rgb) / 2
    return result


def average(img: np.array):
    result = img.copy()
    result[:] = np.sum(img, axis=2, keepdims=True) / 3
    return result


def luminosity(img: np.array):
    red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    result = red*0.21 + green*0.72 + blue*0.07
    return result


def run():
    url = "https://drive.google.com/uc?id=1i9dqan21DjQoG5Q_VEvm0LrVwAlXD0vB"
    try:
        gdown.download(url)
    except FileURLRetrievalError:
        print("File cannot download")

    img = mpimg.imread("dog.jpeg")
    plt.imshow(luminosity(img), cmap="gray")
    plt.show()


run()
