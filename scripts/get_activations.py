# Notes:
#   Script to get activation from saved model checkpoints.
#   Run from root directory.
# CiCi Zheng, Sep, 2024

from neuronet.utils import models

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader


import os
import numpy as np
import random
import matplotlib.pyplot as plt

from neuronet.utils import tools

batch_size = 128

param_names = ['optimizer', 'weight_scale', 'train_size', 'output_scale', 'weight_decay']

param_lists = [ 
                [ 'AdamW', 1, 2000, 1, 0.001],
                [ 'AdamW', 5, 2000, 1, 0.001],
                [ 'AdamW', 10, 2000, 1, 0.001],
                [ 'AdamW', 10, 1000, 1, 0.001],
                [ 'AdamW', 10, 5000, 1, 0.001],
                [ 'AdamW', 10, 2000, .5, 0],
                [ 'AdamW', 10, 2000, .1, 0],
                [ 'AdamW', 10, 2000, .001, 0],
            ]

checkpoint_parent_dir = '/data/cici/Geometry/new/checkpoints'
activation_parent_dir = '/data/cici/Geometry/new/activations'


epochs = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000]  
# epochs = [20000, 30000, 40000, 50000] # For some examples we run for longer

# device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

# standard transform for EMNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

test_dataset = torchvision.datasets.EMNIST(root='./data', split='balanced', train=False, download=False, transform=transform)

subset_indices = np.random.choice(len(test_dataset), 10000, replace=False) # it's about 200 samples per class
test_subset = torch.utils.data.Subset(test_dataset, subset_indices)
test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

for param_list in param_lists:
    param_combo = dict(zip(param_names, param_list))
    MLPmodels, activation_dir = tools.get_models(param_combo, checkpoint_parent_dir, activation_parent_dir, epochs, device)
    tools.get_model_activation(MLPmodels, activation_dir, test_dataloader, device, epochs)
