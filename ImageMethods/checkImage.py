from PIL import Image
import numpy as np

def method(filename):
    img = Image.open(filename)
    if img.format not in ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'):
        return np.NAN
    else:
        return filename
