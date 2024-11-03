from neuronet.utils import models, analysis, tools

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader


import os
import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


num_classes = 47
batch_size = 16
subsample_size = 256 
gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)


param_names = ['optimizer', 'weight_scale', 'train_size', 'output_scale', 'weight_decay']

param_lists = [ 
     ['AdamW', 10, 2000, 1,  0.001],
     ['AdamW', 10, 2000, 1,  0],
     ['AdamW', 100, 2000, 1,  0],
     ['AdamW', 500, 2000, 1,  0],
]

checkpoint_parent_dir = '/data/cici/Geometry/new/checkpoints'
epochs = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500] 

# standard transform for EMNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

test_dataset = torchvision.datasets.EMNIST(root='./data', split='balanced', train=False, download=False, transform=transform)
subset_indices = np.random.choice(len(test_dataset), subsample_size, replace=False) 
test_subset = torch.utils.data.Subset(test_dataset, subset_indices)
test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

train_dataset = torchvision.datasets.EMNIST(root='./data', split='balanced', train=True, download=False, transform=transform)
train_subset = torch.utils.data.Subset(train_dataset, range(subsample_size)) # subsample_size should be < 1000 to apply for all networks.
train_dataloader = DataLoader(train_subset, batch_size= batch_size, shuffle=False)


# the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# paths relative to the script's location
output_parent_dir = os.path.join(script_dir, "..", "output/kernel_distance")


def compute_pairwise_distances(models, dataloader):
    n = len(models)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        ntk_i = analysis.compute_ntk(models[i], dataloader, device)
        for j in range(i, n):
            print(i, j)
            if i == j:
                distance_matrix[i, j] = 0
            else:
                ntk_j = analysis.compute_ntk(models[j], dataloader, device)
                distance = analysis.kernel_distance(ntk_i, ntk_j)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    return distance_matrix

def plot_heatmap(distance_matrix, param_list, save_path):

    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, annot=False, cmap='viridis')
    plt.title(f'Kernel Distance Heatmap for {param_list}')
    plt.xlabel('Model Checkpoint')
    plt.ylabel('Model Checkpoint')
    plt.savefig(save_path)
    plt.close()

# main:
for param_list in param_lists:
    print()
    print('inspecting', param_list, '\n')
    param_combo = dict(zip(param_names, param_list))

    MLPmodels, output_dir = tools.get_models(param_combo, checkpoint_parent_dir, output_parent_dir, epochs, device)

    output_scale = param_combo['output_scale']
    weight_scale = param_combo['weight_scale']

    # choose model type and output scale:
    model0 = models.MLP(num_classes, output_scale=output_scale).to(device)
    # scale weights:
    with torch.no_grad():
        for param in model0.parameters():
            param.data *= weight_scale

    MLPmodels = [model0] + MLPmodels

    # for test set:
    distance_matrix = compute_pairwise_distances(MLPmodels, test_dataloader)
    figname = os.path.join(output_dir, 'kernel_dist_test-set.pdf')
    plot_heatmap(distance_matrix, param_list, figname)
    filename = os.path.join(output_dir, f'K-dist_test.npy')
    np.save(filename, distance_matrix)
    print(f"Data saved as {filename}")

    # for train set:
    distance_matrix = compute_pairwise_distances(MLPmodels, train_dataloader)
    figname = os.path.join(output_dir, 'kernel_dist_train-set.pdf')
    plot_heatmap(distance_matrix, param_list, figname)
    filename = os.path.join(output_dir, f'K-dist_train.npy')
    np.save(filename, distance_matrix)
    print(f"Data saved as {filename}")


  

