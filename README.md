# ArtEvo

Neural style transfer to generate an image by combining the content of one image with the style of another image.

VGG19 model pre-trained on ImageNet, VGG19 model fine-tuned on WikiArt, and lightweight CNN models.

PyTorch tutorial:
https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

HuggingFace WikiArt dataset:
https://huggingface.co/datasets/huggan/wikiart

## Some commands
- pip install torchvision

## Project Scripts
### visualizeInput.py
Load and show the content and style images.

### imageClassifier.py
Preprocess and classify image using pre-trained VGG19.
