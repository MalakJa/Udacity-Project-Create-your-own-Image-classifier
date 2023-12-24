import torch
from torchvision import transforms, datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import numpy as np
import json

import argparse
import utils

parser = argparse.ArgumentParser(description = 'Training script')

parser.add_argument('data_dir', help = 'Input data directory. Mandatory', type=str)
parser.add_argument('--save_dir', help = 'Input saving directory. Optional',  type=str)
parser.add_argument('--arch', help = 'Default is Alexnet, use of VGG16 is possible', type=str, default='alexnet')
parser.add_argument('--learning_rate', help = 'Learning rate. Default is 0.001', type = float, default = 0.001)
parser.add_argument('--dropout', help="Probability value for dropout. Default is 0.3", type=float, default=0.3)
parser.add_argument('--hidden_units', help = 'Hidden units. Default val 2048', type = int, default = 2048)
parser.add_argument('--epochs', help = 'Epochs as integer - default is 8', type = int, default = 8)
parser.add_argument('--gpu', help = "Input gpu to use", type = str)



args = parser.parse_args()
data_dir = args.data_dir

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

save_dir = args.save_dir
lr = args.learning_rate
structure = args.arch
hidden_units = args.hidden_units
gpu = args.gpu
epochs = args.epochs
dropout = args.dropout

if torch.cuda.is_available() and gpu == "gpu":
    device = torch.device("cuda")
    print("You enabled GPU")
else:
    device = torch.device("cpu")


    
training_transforms = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

data_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

#Load the datasets with ImageFolder
image_datasets = [datasets.ImageFolder(train_dir, transform=training_transforms),
                  datasets.ImageFolder(valid_dir, transform=data_transforms),
                  datasets.ImageFolder(test_dir, transform=data_transforms)]

#Using the image datasets and the trainforms, define the dataloaders
dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
               torch.utils.data.DataLoader(image_datasets[1], batch_size=64),
               torch.utils.data.DataLoader(image_datasets[2], batch_size=64)]

#Label mapping
with open('ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
if structure == 'vgg16':
    model = models.vgg16()
    input_layer = 25088
else:
    model = models.alexnet()
    input_layer = 9216
    
for param in model.parameters():
    param.requires_grad = False

    
classifier = nn.Sequential(
                            nn.Linear(input_layer, 4096),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Linear(4096, hidden_units),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Linear(hidden_units, 102),
                            nn.LogSoftmax(dim=1)
                            )
model.classifier = classifier

# initializing criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)

def validating(model, validation_set, criterion):
    model.to(device)
    
    validation_loss = 0
    accuracy = 0
    
    for validation_inputs, validation_labels in validation_set:
        validation_inputs, validation_labels = validation_inputs.to(device), validation_labels.to(device)
        output = model.forward(validation_inputs)
        loss = criterion(output, validation_labels).item()
        validation_loss += loss

        ps = torch.exp(output)
        equality = (validation_labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return validation_loss, accuracy


model.to(device)
epochs = epochs
print_every = 30
steps = 0
for epoch in range(epochs):
    running_loss = 0
    for inputs, labels in dataloaders[0]:
        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)

        
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        if steps % print_every == 0:
            model.eval()
            
            with torch.no_grad():
                validation_loss, accuracy = validating(model, dataloaders[1], criterion)
            
            print("Epoch: {} out of {}: ".format(epoch + 1, epochs),
                  "Training Loss: {:.3f} | ".format(running_loss/print_every),
                  "Validation Loss: {:.3f} | ".format(validation_loss/len(dataloaders[1])),
                  "Validation Accuracy: {:.3f}%".format((accuracy/len(dataloaders[1]))*100))

            running_loss = 0       
            model.train()

model.class_to_idx = image_datasets[0].class_to_idx
checkpoint = {'classifier': model.classifier, 
              'state_dict': model.state_dict(), 
              'class_to_idx': model.class_to_idx,
              'arch': args.arch} 

if save_dir:
    torch.save(checkpoint, save_dir + '/checkpoint.pth')
else:
    torch.save(checkpoint, 'ImageClassifier/checkpoint.pth') 