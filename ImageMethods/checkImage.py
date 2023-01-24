from PIL import Image
import numpy as np

def method(filename):
    valid = ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')

    try:
        img = Image.open(filename)
        return np.NaN if img.format not in valid else filename
    except IOError:
        return np.NaN

