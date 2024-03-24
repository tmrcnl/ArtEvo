import torch

import lib.constants as constants
import lib.image as image


print('content file (original): ', constants.CONTENT_FILE)
content_image = image.load_img(constants.CONTENT_FILE)
image.show_img(content_image)

print('content file (preprocessed):')
content_image_preprocessed = image.preprocess_img(content_image, constants.IMG_SIZE)
image.show_img(content_image_preprocessed)

print('style file: ', constants.STYLE_FILE)
style_image = image.load_img(constants.STYLE_FILE)
image.show_img(style_image)

print('style file (preprocessed):')
style_image_preprocessed = image.preprocess_img(style_image, constants.IMG_SIZE)
image.show_img(style_image_preprocessed)
