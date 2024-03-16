import matplotlib.pyplot as plt
import numpy as np

import torchvision

def load_img(path):

    # read image at path and return as a tensor
    img = torchvision.io.read_image(path)
    return img


def show_img(img):

    # show image
    npimg = img.numpy()    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()