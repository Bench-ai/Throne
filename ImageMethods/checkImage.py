from PIL import Image
import numpy as np

def method(filename):
    global img
    valid = ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')

    try:
        img = Image.open(filename)
    except IOError:
        return np.NaN

    return np.NaN if img.format not in valid else filename
