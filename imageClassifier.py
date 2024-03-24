import torch

from torchvision.models import vgg19, VGG19_Weights

import torch.nn.functional as F

import lib.constants as constants
import lib.image as image

import json

# load mapping of index to label for imagenet1000 image ckassification
class_idx = json.load(open(constants.CLASS_MAP))
idx2label = [class_idx[str(i)][1] for i in range(len(class_idx))]

# load content image
content_image = image.load_img(constants.CONTENT_FILE)

# preprocess content image for VGG 19
content_image_preprocessed = image.preprocess_img(content_image, constants.IMG_SIZE)

# instantiate pre-trained VGG19 model, in evaluation mode
# cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
cnn = vgg19(weights=VGG19_Weights.DEFAULT)

# y predictions
y_pred = cnn(content_image_preprocessed)

# y probabilities
y_prob = F.softmax(y_pred)

# predicted index
pred_idx = torch.argmax(y_prob, 1)

# predicted label (from mapping)
pred_label = idx2label[pred_idx]

# print('y_pred: ', y_pred)
# print('y_prob: ', y_prob)
# print('y_prob.shape: ', y_prob.shape)
print('pred_idx: ', pred_idx)
print('pred_label: ', pred_label)