from neuronet.utils import models

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import random
import numpy as np

# Set device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# standard transform for EMNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.EMNIST(root='../data', split='balanced', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.EMNIST(root='../data', split='balanced', train=False, download=False, transform=transform)

# Set seeds
seed = 314159
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Set the random seed for Python and NumPy, just in case
random.seed(seed)
np.random.seed(seed)

num_epochs = 10000

param_names = ['train_size', 'weight_scale','weight_decay', 'output_scale', 'optimizer']
param_vals = [1000, 10,  0.001, 1, 'Adam']
params = dict(zip(param_names, param_vals))

checkpoint_parent_dir = '/data/cici/Geometry/new/checkpoints'
checkpoint_dir = os.path.join(checkpoint_parent_dir, '_'.join([f"{param_name}:{value}" for param_name, value in params.items()]))
os.makedirs(checkpoint_dir, exist_ok=True)


### Main function:
def main(params, num_epochs, checkpoint_dir):

    num_classes = 47
    
    train_size = params['train_size']
    optimizer_name = params['optimizer']
    output_scale = params['output_scale']
    weight_decay = params['weight_decay']
    weight_scale = params['weight_scale']

    loss_function = 'CrossEntropy'
    learning_rate = 0.001

    # choose model type and output scale:
    model = models.MLP(num_classes, output_scale=output_scale).to(device)
    
    # subset training sample size:
    train_subset = torch.utils.data.Subset(train_dataset, range(train_size))

    train_loader = DataLoader(train_subset, batch_size= 128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size= 128, shuffle=False)

    # scale weights:
    with torch.no_grad():
        for param in model.parameters():
            param.data *= weight_scale
    
    # choose loss func:
    if loss_function == 'MSE':
        criterion = nn.MSELoss()
    elif loss_function == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid loss function: {loss_function}")
    
    # choose optimizer
    optimizer_dict = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    }
    optimizer_class = optimizer_dict.get(optimizer_name)

    # set optimizer with weight deccay:
    optimizer = optimizer_class(model.parameters(), lr= learning_rate, weight_decay=weight_decay)

    # train:
    train_accuracies, test_accuracies, train_losses, test_losses = models.train(model, train_loader, test_loader, criterion, optimizer, device, num_epochs, checkpoint_dir)
    
    # title = f"MLP,OutputScale:{output_scale},TrainingSize:{train_size},Optimizer:{optimizer_name},WeightDecay:{weight_decay}"

    title = f'{optimizer_name}-{train_size}-{output_scale}-{weight_decay}'
    
    models.visualize(train_accuracies, test_accuracies, train_losses, test_losses, title)

if __name__ == "__main__":
    main(params, num_epochs, checkpoint_dir)