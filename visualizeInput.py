import torch

import lib.constants as constants
import lib.showimage as showimage


print('content file: ', constants.CONTENT_FILE)
content_image = showimage.load_img(constants.CONTENT_FILE)
showimage.show_img(content_image)

print('style file: ', constants.STYLE_FILE)
style_image = showimage.load_img(constants.STYLE_FILE)
showimage.show_img(style_image)
