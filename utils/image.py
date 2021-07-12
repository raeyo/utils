from PIL import Image
import numpy as np

def save_numpy_to_image(ndarray, path):
  if ndarray.dtype == np.uint8:
    rgb_image = Image.fromarray(ndarray)
  else:
    rgb_image = Image.fromarray(np.uint8(ndarray * 255))
  rgb_image.save(path)