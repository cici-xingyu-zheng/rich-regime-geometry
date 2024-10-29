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

num_classes = 10

def get_models(param_combo, checkpoint_parent_dir, output_parent_dir, epochs, device):

    subdir = '_'.join([f"{param_name}:{value}" for param_name, value in param_combo.items()])
    checkpoint_dir = os.path.join(checkpoint_parent_dir, subdir)
    output_dir = os.path.join(output_parent_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    # Load trained networks:
    MLPmodels = []
    for epoch in epochs:
        checkpoint = torch.load(f'{checkpoint_dir}/model_epoch_{epoch}.pth') ## check name
        model = create_mlp(alpha = param_combo['alpha']).to(device)
        model.load_state_dict(checkpoint)
        MLPmodels.append(model)

    return MLPmodels, output_dir


def create_mlp(alpha=1.0):
    # if it is weight scale, its already in the checkpoints!
    """Creates an MLP model with specified depth, width, activation, and output scaling."""
    width = 200
    depth = 3
    layers = [nn.Flatten()]
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(28 * 28, width))
            layers.append(nn.ReLU())
        elif i == depth - 1:
            layers.append(nn.Linear(width, 10))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

    class OutputScaledMLP(nn.Module):
        def __init__(self, mlp, alpha):
            super().__init__()
            self.mlp = mlp
            self.alpha = alpha

        def forward(self, x):
            return self.alpha * self.mlp(x)

    return OutputScaledMLP(nn.Sequential(*layers), alpha).to(device)


batch_size = 200

epochs = [1, 10, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
# device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

download_directory = "../data"
checkpoint_parent_dir = '/data/cici/Geometry/new/MNIST/checkpoints'
activation_parent_dir = '/data/cici/Geometry/new/MNIST/activations'

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

param_names = ['type', 'alpha', 'scale']

param_lists = [ 
                [ 'alpha', 0.001, 1],
                [ 'alpha', 0.5, 1],
                [ 'scale', 1, 2],
                [ 'scale', 1, 8],
            ]

# load dataset
test_dataset = torchvision.datasets.MNIST(root=download_directory, train=False,
    transform=torchvision.transforms.ToTensor(), download=False)
subset_indices = np.random.choice(len(test_dataset), 2000, replace=False) # it's about 200 samples per class
test_subset = torch.utils.data.Subset(test_dataset, subset_indices)
test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)


for param_list in param_lists:
    param_combo = dict(zip(param_names, param_list))
    MLPmodels, activation_dir = tools.get_models(param_combo, checkpoint_parent_dir, activation_parent_dir, epochs, device)
    tools.get_model_activation(MLPmodels, activation_dir, test_dataloader, device, epochs)
