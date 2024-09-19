from collections import defaultdict
from itertools import islice
import random
import time
import os
from pathlib import Path
import math

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt
import seaborn as sns

train_points = 512 # 256 was used for the paper 
optimization_steps = 100001
batch_size = 16
loss_function = 'MSE'   # 'MSE' or 'CrossEntropy'
optimizer = 'AdamW'     # 'AdamW' or 'Adam' or 'SGD'
lr = 1e-3
initialization_scale = 8.0
download_directory = "../data"
weight_decay = 0
depth = 3              # the number of nn.Linear modules in the model
width = 200
activation = 'ReLU'     # 'ReLU' or 'Tanh' or 'Sigmoid' or 'GELU'

optimizer_dict = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
}

activation_dict = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU
}

loss_function_dict = {
    'MSE': nn.MSELoss,
    'CrossEntropy': nn.CrossEntropyLoss
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
seed = 0

activation_fn = activation_dict[activation]

def create_mlp(depth, width, activation, alpha=1.0):
    """Creates an MLP model with specified depth, width, activation, and output scaling."""
    layers = [nn.Flatten()]
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(28 * 28, width))
            layers.append(activation_fn())
        elif i == depth - 1:
            layers.append(nn.Linear(width, 10))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation_fn())

    class OutputScaledMLP(nn.Module):
        def __init__(self, mlp, alpha):
            super().__init__()
            self.mlp = mlp
            self.alpha = alpha

        def forward(self, x):
            return self.alpha * self.mlp(x)

    return OutputScaledMLP(nn.Sequential(*layers), alpha).to(device)

# load dataset
train = torchvision.datasets.MNIST(root=download_directory, train=True,
    transform=torchvision.transforms.ToTensor(), download=False)
test = torchvision.datasets.MNIST(root=download_directory, train=False,
    transform=torchvision.transforms.ToTensor(), download=False)
train = torch.utils.data.Subset(train, range(train_points))
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

subset_indices = np.random.choice(len(test), train_points, replace=False) 
test_subset = torch.utils.data.Subset(test, subset_indices)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)

def kernel_distance(Kt1, Kt2):
    """
    Compute the kernel distance between two Neural Tangent Kernels using PyTorch.
    
    Args:
        Kt1: NTK matrix at time t1 (PyTorch tensor)
        Kt2: NTK matrix at time t2 (PyTorch tensor)
    
    Returns:
        The kernel distance between Kt1 and Kt2 (PyTorch tensor)
    """
    # Ensure inputs are on the same device
    if Kt1.device != Kt2.device:
        Kt2 = Kt2.to(Kt1.device)

    # Compute the Frobenius inner product
    frobenius_inner_product = torch.sum(Kt1 * Kt2)
    
    # Compute the Frobenius norms
    frobenius_norm_Kt1 = torch.sqrt(torch.sum(Kt1**2))
    frobenius_norm_Kt2 = torch.sqrt(torch.sum(Kt2**2))
    
    # Compute the kernel distance
    kernel_dist = 1 - frobenius_inner_product / (frobenius_norm_Kt1 * frobenius_norm_Kt2)
    
    return kernel_dist

def compute_ntk(model, dataloader, device):
    '''
    Compute the Neural Tangent Kernel (NTK) for the given model and dataset.
    Most computations are performed on CPU to reduce GPU memory usage..
    '''
    model.eval()  
    for param in model.parameters():
        param.requires_grad = True

    all_jacobians = []

    for batch_idx, batch in enumerate(dataloader):
        # print(f"Processing batch {batch_idx + 1}")
        
        # handle different input types
        if isinstance(batch, list):
            inputs = batch[0]  # Assume the first element is the input data
        else:
            inputs = batch  # [batch_size, input_size]
        
        # convert to tensor if it's not already
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        
        inputs = inputs.to(device)
        # print(f"Batch {batch_idx + 1}: Input shape = {inputs.shape}")
        
        model.zero_grad()
        outputs = model(inputs)
        # print(f"Batch {batch_idx + 1}: Output shape = {outputs.shape}")
        
        # move outputs to CPU for further processing, probably cost computation time..
        outputs = outputs.cpu()
        
        batch_jacobians = []
        for output in outputs:
            jacobian_rows = []
            for out_idx in range(output.size(0)):
                grad_output = torch.zeros_like(output)
                grad_output[out_idx] = 1.0
                gradients = torch.autograd.grad(output.to(device), model.parameters(), grad_outputs=grad_output.to(device), retain_graph=True)
                jacobian_row = torch.cat([g.cpu().flatten() for g in gradients])
                jacobian_rows.append(jacobian_row)
            jacobian = torch.stack(jacobian_rows)
            batch_jacobians.append(jacobian)
        
        batch_jacobian = torch.cat(batch_jacobians, dim=0)
        all_jacobians.append(batch_jacobian)
        
        # print(f"Batch {batch_idx + 1}: Processed {batch_jacobian.shape[0]} inputs, Jacobian shape = {batch_jacobian.shape}")
        
        # free up memory
        del batch_jacobians, jacobian_rows, gradients, grad_output, outputs
        torch.cuda.empty_cache()  # If using GPU

    # Concatenate all Jacobians on CPU
    full_jacobian = torch.cat(all_jacobians, dim=0)
    print(f"Full Jacobian shape: {full_jacobian.shape}")

    # Compute the NTK on CPU
    ntk = torch.mm(full_jacobian, full_jacobian.t())
    print(f"NTK shape: {ntk.shape}")

    print("NTK computation completed.")
    return ntk

# Helper function to estimate memory usage
def estimate_memory_usage(tensor):
    return tensor.element_size() * tensor.nelement() / 1e9  # in GB


def compute_pairwise_distances(models, dataloader):
    n = len(models)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        ntk_i = compute_ntk(models[i], dataloader, device)
        for j in range(i, n):
            print(i, j)
            if i == j:
                distance_matrix[i, j] = 0
            else:
                ntk_j = compute_ntk(models[j], dataloader, device)
                distance = kernel_distance(ntk_i, ntk_j)
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


alphas = [0.5, 0.001]
alpha = alphas[0]

steps = [1, 10, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

for alpha in alphas:
    print(alpha)
    MLPmodels = []
    mlp0 = create_mlp(depth, width, activation_fn, .5)
    with torch.no_grad():
        for p in mlp0.parameters():
            p.data = initialization_scale * p.data

    MLPmodels.append(mlp0)
    for step in steps:
        checkpoint = torch.load(os.path.join('/data/cici/Geometry/new/MNIST/checkpoints', f'alpha_{alpha}_{step}.pth'))
        mlp = create_mlp(depth, width, activation_fn, .5)
        mlp.load_state_dict(checkpoint)
        MLPmodels.append(mlp)

    distance_matrix = compute_pairwise_distances(MLPmodels, test_loader)
    plot_heatmap(distance_matrix, f'MNIST, alpha = {alpha}',f'./K-dist_{alpha}_test_512.pdf')
    np.save(f'K-dist_{alpha}_test_512.npy', distance_matrix)
    distance_matrix = compute_pairwise_distances(MLPmodels, train_loader)
    plot_heatmap(distance_matrix, f'MNIST, alpha = {alpha}',f'./K-dist_{alpha}_train_512.pdf')
    np.save(f'K-dist_{alpha}_train_512.npy', distance_matrix)
