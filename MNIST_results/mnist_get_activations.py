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

def get_models(param_list, checkpoint_parent_dir, output_parent_dir, epochs, device):

    subdir = '_'.join([param for param in param_list])
    output_dir = os.path.join(output_parent_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    # Load trained networks:
    MLPmodels = []
    for epoch in epochs:
        checkpoint = torch.load(f'{checkpoint_parent_dir}/{param_list[0]}_{param_list[1]}_{epoch}.pth') ## check name
        if param_list[0] == "scale":
            model = create_mlp().to(device)
        else:
            # set output scaling
            model = create_mlp(alpha = float(param_list[1]))
        model.load_state_dict(checkpoint)
        MLPmodels.append(model)

    return MLPmodels, output_dir


def create_mlp(alpha=1.0):
    """Creates an MLP model matching the trained model, to load checkpoints; 
    specify output scaling."""
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

param_lists = [ 
                [ 'alpha', '0.001'],
                [ 'alpha', '0.5'],
                [ 'scale', '2'],
                [ 'scale', '8'],
            ]

# load dataset
test_dataset = torchvision.datasets.MNIST(root=download_directory, train=False,
    transform=torchvision.transforms.ToTensor(), download=True)
subset_indices = np.random.choice(len(test_dataset), 2000, replace=False) # it's about 200 samples per class
test_subset = torch.utils.data.Subset(test_dataset, subset_indices)
test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)


for param_list in param_lists:
    MLPmodels, activation_dir = get_models(param_list, checkpoint_parent_dir, activation_parent_dir, epochs, device)
    tools.get_model_activation(MLPmodels, activation_dir, test_dataloader, device, epochs)
