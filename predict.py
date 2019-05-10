# import libraries
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
import numpy as np
import seaborn as sb
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str, help = 'The path of Images')
parser.add_argument('checkpoint', type=str, help = 'Model checkpoint')
parser.add_argument('--top_k', type=int, default=5, help = 'Show top k predictions')
parser.add_argument('--category_names', type=str, default="cat_to_name.json", help = 'category file')
parser.add_argument('--gpu', default=False, help = 'Using or not using GPU')
parser = parser.parse_args()

with open(parser.category_names, 'r') as f:
    cat_to_name = json.load(f)

if parser.gpu and torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')


def load_model(path):
    checkpoint = torch.load(path)
    model = models.densenet161(pretrained=True)
    model.classifier = checkpoint['classifier']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    image = transform(pil_image)
    return image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    img = process_image(image_path)
    img.unsqueeze_(0)
    model.eval()
    model.cpu()
    with torch.no_grad():
        logps = model.forward(img)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p = top_p.numpy().tolist()[0]
        top_class = top_class.numpy().tolist()[0]
        
        # convert to index
        index_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    classes = []
    for n_class in top_class:
        classes.append(index_to_class[n_class])
    
    return top_p, classes

def main():
    model = load_model(parser.checkpoint)
    top_probs, chosen_classes = predict(parser.image_path, model, parser.top_k)
    #flower_num = image_path.split('/')[2]
    image = process_image(parser.image_path)
    # predict
    probs, classes = predict(parser.image_path, model, topk=parser.top_k)
    flower_name = []
    for class_idx in classes:
        flower_name.append(cat_to_name[class_idx])
    for itr in zip(flower_name, top_probs):
        print('The prediction is {}. The probability of it is {:.2f}'.format(itr[0],itr[1]))

    title =  parser.image_path.split('/')[2]
    real_name = cat_to_name[title]
    print('The real name is {}'.format(real_name))

if __name__ == '__main__':
    
    main()