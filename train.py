#import libraries
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

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='flowers', help='Dataset path')
parser.add_argument('--gpu', action='store_true', help='Use GPU')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--arch', type=str, default='densenet161', help='Model architecture')
parser.add_argument('--output', type=int, default=102, help='Number of output')
parser.add_argument('--learning_rate', type=float, default=0.0015, help='Learning rate')
parser.add_argument('--hidden_units_1', type=int, default=512, help='Number of first hidden units')
parser.add_argument('--hidden_units_2', type=int, default=256, help='Number of second hidden units')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Save trained model')

args, _ = parser.parse_known_args()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# transform image
def transform_image(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle = False)
    
    class_to_idx = train_data.class_to_idx
    
    return trainloader, validloader, testloader, class_to_idx
    
    
# create model
def set_model(arch = 'densenet161', output = 102, hidden_units_1 = 512, hidden_units_2 = 256, learning_rate = 0.0015):
    trainloader, validloader, testloader, class_to_idx = transform_image(args.data_dir)
    if arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        in_features = model.classifier.in_features
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[0].in_features
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
    else:
        print('Unexpected architecture')
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(in_features, hidden_units_1)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(0.3)),
                            ('fc2', nn.Linear(hidden_units_1, hidden_units_2)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(0.2)),
                            ('fc3', nn.Linear(hidden_units_2, output)),
                            ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.class_to_idx = class_to_idx
    return model, optimizer, criterion, learning_rate


def train_model(epochs):
    trainloader, validloader, testloader, class_to_idx = transform_image(args.data_dir)
    model, optimizer, criterion, learning_rate = set_model(args.arch, args.output, args.hidden_units_1, args.hidden_units_2, args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 40
    for epoch in range(epochs):

        for inputs, labels in iter(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                accuracy = 0
                valid_loss = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in iter(validloader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    return model, optimizer, epochs, learning_rate

def save_model(name = 'checkpoint.pth'):
    model, optimizer, epochs, learning_rate = train_model(args.epochs)
    checkpoint = {'epochs': epochs,
                  'learning_rate': learning_rate,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()
                 }

    torch.save(checkpoint, name)

def load_model(path):
    checkpoint = torch.load(path)
    model = models.densenet161(pretrained=True)
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
def main():
    save_model(args.save_dir)

if __name__ == '__main__':
    main()
