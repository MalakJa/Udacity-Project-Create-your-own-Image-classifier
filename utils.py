import torch
from PIL import Image
import numpy as np
import json
import torchvision.models as models

def load_model(path):
    checkpoint = torch.load(path) 
    model = models.alexnet() 

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    image = Image.open(image)
    
    # "First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio"
    if image.size[0] > image.size[1]:
        size = [image.size[0], 256]
    else:
        size = [256, image.size[1]]
    
    image.thumbnail(size)
    
    # "Then you'll need to crop out the center 224x224 portion of the image"
    left = (256 - 224)/2
    top = (256 - 224)/2
    right = (256 + 224)/2
    bottom = (256 + 224)/2

    image = image.crop((left, top, right, bottom))
    
    # "Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1."
    image = np.array(image)
    image = image / 255
    
    # "You'll want to subtract the means from each color channel, then divide by the standard deviation"
    mean = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
                       
    image = ((image - mean) / stds)
    
    
    # "PyTorch expects the color channel to be the first dimension "
    image = np.transpose(image, (2, 0, 1))
    
    return image


def load_category_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names