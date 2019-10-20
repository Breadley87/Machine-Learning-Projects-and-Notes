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
import pickle

# define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser(description = "Parser for prediction arguments")

parser.add_argument('image_dir', help = 'path to image used for prediction. Required', type = str)
parser.add_argument('load_dir', help = 'Path to trained model. Required', type = str)
parser.add_argument('--top_k', help = 'NUmber of prediction classes and probabilities to be displayed. Not required, default 5.', type = int)
parser.add_argument('--category_names', help = 'Mapping of prediction categories to the real name. Not required', type = str)
parser.add_argument('--gpu', help = "turn on gpu mode. Not required", type = str)

args = parser.parse_args()
file_path = args.image_dir

# Setting to GPU if arg is chosen
if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'
    
    # Load file with category names, if one is selected
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

# Setting the number of predictions to display, default to the highest probability prediciton.
if args.top_k:
    class_n = args.top_k
else:
    class_n = 1


def loading_model(file_path):
    saved_model = torch.load(file_path) #loading checkpoint from a file
    if saved_model['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.alexnet(pretrained=True)
        
    model.classifier = saved_model['classifier']
    model.load_state_dict(saved_model['state_dict'])
    model.class_to_idx = saved_model['mapping']
    
    for param in model.parameters(): 
        param.requires_grad = False #turning off tuning of the model
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Converting image to PIL image using image file path
    pil_image = Image.open(f'{image}' + '.jpg')

    # Building image transform
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    
    ## Transforming image for use with network
    pil_transform = transform(pil_image)
    
    # Converting to Numpy array 
    pil_array = np.array(pil_transform)
    
    return pil_array

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Pre-processing image
    img = process_image(image_path)
    # Converting to torch tensor from Numpy array
    if device == 'cuda':
        image_tensor = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    else:
        image_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = image_tensor.unsqueeze_(0)
    
    #Enable GPU or CPU
    model.to(device)
    img_add_dim.to(device)


    # Setting model to evaluation mode and turning off gradients
    model.eval()
    with torch.no_grad():
        # Running image through network
        output = model.forward(img_add_dim)

    # Calculating probabilities
    ps = torch.exp(output)
    top_ps = ps.topk(topk)[0]
    index = ps.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    top_ps_list = np.array(top_ps)[0]
    top_index_list = np.array(index[0])
    
    # Loading index and class mapping
    class_to_idx = model.class_to_idx
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes = []
    for i in top_index_list:
        classes += [indx_to_class[i]]
        
    return top_ps_list, classes

# Load saved model for prediction
model = loading_model(args.load_dir)

ps, classes = predict(file_path, model, class_n, device)

# Defining class names for prediction if loaded
class_names = [cat_to_name[i] for i in classes]

for x in range(class_n):
     print("The model suggests that the given image has the class name: {} ".format(class_names[x]),
            "with a probability of: {:.2f}% ".format(ps[x]*100),)

