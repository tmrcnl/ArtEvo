import os

FILES_DIR = 'files'

# https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg
CONTENT_FILE = os.path.join(FILES_DIR, 'YellowLabradorLooking_new.jpg')

# https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg
STYLE_FILE = os.path.join(FILES_DIR, 'Vassily_Kandinsky,_1913_-_Composition_7.jpg')

# image size (use 128x128 for local CPU processing)
IMG_SIZE = (128, 128)

# imagenet class index <-> name mapping
CLASS_MAP = os.path.join(FILES_DIR, 'imagenet_class_index.json')