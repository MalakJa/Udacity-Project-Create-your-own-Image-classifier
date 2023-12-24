import numpy as np
import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable

import matplotlib.pyplot as plt
import json
from PIL import Image

import utils

parser = argparse.ArgumentParser(description = 'Predicting script')
parser.add_argument('image_dir', help = 'Input image path. Mandatory', type = str)
parser.add_argument('--checkpoint', help = 'Checkpoint path. Mandatory', default = "ImageClassifier/checkpoint.pth", type = str)
parser.add_argument('--top_k', help = 'Choose number of Top K classes. Default is 5', default = 5, type = int)
parser.add_argument('--gpu', help = "Input gpu to use", type = str)
parser.add_argument('--category_names', help = 'Path of JSON file mapping categories to names. Optional', default='ImageClassifier/cat_to_name.json', type = str)


args = parser.parse_args()

image_dir = args.image_dir
checkpoint = args.checkpoint
category_names = args.category_names
top_k = args.top_k
gpu = args.gpu


if gpu == "gpu":
    device = 'cuda'
else:
    device = 'cpu'
 

model = utils.load_model(checkpoint) 
    
for param in model.parameters():
    param.requires_grad = False
    
  
category_labels = utils.load_category_names(category_names)
    
def predict(image_path, model, topk=top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()
    img = utils.process_image(image_path)
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        outputs = model.forward(img.to(device))
        
    probability = torch.exp(outputs)
    
    top_ps, top_classes = probability.topk(topk, dim=1)       
    top_p = top_ps.tolist()[0]
    
    classes = {val:category_labels[k] for k, val in model.class_to_idx.items()}  

    top_classes = [classes[i] for i in top_classes.tolist()[0]]

    for i in range(len(top_p)):
        print("Predicted {} with probability of {:0.3f}".format(top_classes[i],top_p[i]))
    


predict(image_dir, model)


