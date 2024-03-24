import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.io as io
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional

from PIL import Image


def load_img(path):

    # # read image at path and return as a tensor
    # img = io.read_image(path)

    # load PIL image from path
    PIL_img = Image.open(path)

    # convert PIL_img to tensor
    img = functional.to_tensor(PIL_img)

    return img


def preprocess_img(img, img_size):

    # image resize to img_size
    resize = transforms.Resize(img_size)

    # image normalization pipeline (these values are for VGG19)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # instantiate image preprocessing pipeline
    preprocess = transforms.Compose([resize, normalize])

    # preprocess img
    img = preprocess(img).unsqueeze(0)

    return img


def show_img(img):

    img = img.cpu().clone()  # clone image
    img = img.squeeze(0) 

    # convert tensor image to PIL 
    PIL_img = functional.to_pil_image(img)

    # show image
    plt.imshow(PIL_img)
    plt.show()

    # # show image
    # npimg = img.numpy()    
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()

    