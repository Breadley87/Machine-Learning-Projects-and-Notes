#Imports
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data 
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

# define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser(description = "Parser for training arguments")

parser.add_argument('data_dir', help = 'Data Directory is a required arg', type = str)
parser.add_argument('--save_dir', help = 'Directory for saving results. Arg is optional', type = str)
parser.add_argument('--arch', help = 'Vgg13 is the default model type but Alexnet can also be used if specified', type = str)
parser.add_argument('--lr', help = 'Optional learning rate arg, default is 0.0005', type = float)
parser.add_argument('--hidden_units', help = 'Optional setting for hidden units in classifier. Default value is 2048', type = int)
parser.add_argument('--epochs', help = 'Number of epochs for training, default is 5.', type = int)
parser.add_argument('--gpu', help = "This setting allows the use of GPU", type = str)

#setting values data loading
args = parser.parse_args()

#setting image paths
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#setting device type from input args
if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'
    
means = [0.485, 0.456, 0.406]
stdevs = [0.229, 0.224, 0.225]

if data_dir:
    train_data_transforms = transforms.Compose([transforms.RandomRotation((-180, 180)),
                                                 transforms.RandomResizedCrop(size=224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(means, stdevs)])

    validation_data_transforms = transforms.Compose([transforms.Resize(size=255),
                                                    transforms.CenterCrop(size=224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(means, stdevs)])

    test_data_transforms = transforms.Compose([transforms.Resize(size=255),
                                              transforms.CenterCrop(size=224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stdevs)])

    # Load the datasets with ImageFolder
    train_images = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    validation_images = datasets.ImageFolder(valid_dir, transform=validation_data_transforms)
    test_images = datasets.ImageFolder(test_dir, transform=test_data_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=32, shuffle=True)

#mapping from category label to category name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model(arch, hidden_units):
    if arch == 'alexnet': #setting model based on vgg13
        model = models.alexnet(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units: #in case hidden_units were given
            classifier = nn.Sequential (OrderedDict ([
                            ('fc1', nn.Linear(9216, 4096)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p = 0.2)),
                            ('fc2', nn.Linear(4096, hidden_units)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p = 0.2)),
                            ('fc3', nn.Linear(hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim =1))
                            ]))
        else: #if hidden_units not given
            classifier = nn.Sequential(OrderedDict ([
                        ('fc1', nn.Linear(9216, 4096)),
                        ('relu1', nn.ReLU()),
                        ('dropout1', nn.Dropout(p = 0.2)),
                        ('fc2', nn.Linear(4096, 2048)),
                        ('relu2', nn.ReLU()),
                        ('dropout2', nn.Dropout(p = 0.2)),
                        ('fc3', nn.Linear(2048, 102)),
                        ('output', nn.LogSoftmax(dim =1))
                        ]))
        model.classifier = classifier     
    else: #setting model to default vgg13
        arch = 'vgg13' #will be used for checkpoint saving, so should be explicitly defined
        model = models.vgg13(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units: 
            classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(25088, 9216)),
                        ('relu1', nn.ReLU()),
                        ('dropout1', nn.Dropout(p = 0.2)),
                        ('fc2', nn.Linear(9216, hidden_units)),
                        ('relu2', nn.ReLU()),
                        ('dropout2', nn.Dropout(p = 0.2)),
                        ('fc3', nn.Linear (hidden_units, 102)),
                        ('output', nn.LogSoftmax(dim=1))]))
        else: #if hidden_units not given
            classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(25088, 9216)),
                        ('relu1', nn.ReLU()),
                        ('dropout1', nn.Dropout(p = 0.2)),
                        ('fc2', nn.Linear(9216, 2048)),
                        ('relu2', nn.ReLU()),
                        ('dropout2', nn.Dropout(p = 0.2)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax(dim=1))]))
            model.classifier = classifier #we can set classifier only once as cluasses self excluding (if/else)
    return model, arch

# Defining validation method 
def validation(model, validation_loader, criterion):
    model.to(device)
    
    valid_loss = 0
    accuracy = 0
    for inputs, labels in validation_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

# model load
model, arch = load_model(args.arch, args.hidden_units)

# model training
criterion = nn.NLLLoss()
if args.lr:
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0005)
    
model.to(device)

if args.epochs:
    epochs = args.epochs
else:
    epochs = 5
print_every = 50
steps = 0


for e in range(epochs): 
    running_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        steps += 1
    
        inputs, labels = inputs.to(device), labels.to(device)
    
        optimizer.zero_grad() #where optimizer is working on classifier paramters only
    
        # Forward and backward passes
        outputs = model.forward(inputs) #calculating output
        loss = criterion(outputs, labels) #calculating loss
        loss.backward() 
        optimizer.step() #performs single optimization step 
    
        running_loss += loss.item() # loss.item() returns scalar value of Loss function
    
        if steps % print_every == 0:
            model.eval() #switching to evaluation mode so that dropout is turned off
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                validation_loss, accuracy = validation(model, validation_loader, criterion)
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(validation_loss/len(validation_loader)),
                  "Validation Accuracy: {:.3f}%".format(accuracy/len(validation_loader)*100))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()

model.to('cpu') #no need to use cuda for saving/loading model.
# TODO: Save the checkpoint 
model.class_to_idx = train_images.class_to_idx #saving mapping between predicted class and class name, 
#second variable is a class name in numeric 

#creating dictionary 
saved_model = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'mapping':    model.class_to_idx,
              'opt_state': optimizer.state_dict,
              'num_epochs': epochs,
              'arch': arch}        
if args.save_dir:
    torch.save(saved_model, args.save_dir + '/project_checkpoint.pth')
else:
    torch.save(saved_model, 'project_checkpoint.pth')