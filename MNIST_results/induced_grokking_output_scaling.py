from collections import defaultdict
from itertools import islice
import random
import time
import os
from pathlib import Path
import math
import pickle as pkl
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt

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

train_points = 1000
optimization_steps = 100001
batch_size = 200
loss_function = 'MSE'   # 'MSE' or 'CrossEntropy'
optimizer_choice = 'AdamW'     # 'AdamW' or 'Adam' or 'SGD'
lr = 1e-3
initialization_scale = 8.0
download_directory = "../data"
weight_decay = 0
depth = 3              # the number of nn.Linear modules in the model
width = 200
activation = 'ReLU'     # 'ReLU' or 'Tanh' or 'Sigmoid' or 'GELU'

log_freq = math.ceil(optimization_steps / 150)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
seed = 0

activation_fn = activation_dict[activation]

def create_mlp(depth, width, activation_fn, alpha=1.0):
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


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def compute_accuracy(network, dataset, device, N=2000, batch_size=50):
    """Computes accuracy of `network` on `dataset`.
    """
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        correct = 0
        total = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            logits = network(x.to(device))
            predicted_labels = torch.argmax(logits, dim=1)
            correct += torch.sum(predicted_labels == labels.to(device))
            total += x.size(0)
        return (correct / total).item()

def compute_loss(network, dataset, loss_function, device, N=2000, batch_size=50):
    """Computes mean loss of `network` on `dataset`.
    """
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_fn = loss_function_dict[loss_function](reduction='sum')
        one_hots = torch.eye(10, 10).to(device)
        total = 0
        points = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            y = network(x.to(device))
            if loss_function == 'CrossEntropy':
                total += loss_fn(y, labels.to(device)).item()
            elif loss_function == 'MSE':
                total += loss_fn(y, one_hots[labels]).item()
            points += len(labels)
        return total / points


# load dataset
train = torchvision.datasets.MNIST(root=download_directory, train=True,
    transform=torchvision.transforms.ToTensor(), download=False)
test = torchvision.datasets.MNIST(root=download_directory, train=False,
    transform=torchvision.transforms.ToTensor(), download=False)
train = torch.utils.data.Subset(train, range(train_points))
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

assert activation in activation_dict, f"Unsupported activation function: {activation}"
activation_fn = activation_dict[activation]

alphas = [0.001, 0.5]

results = {}
for alpha in alphas:

    print(f"alpha: {alpha}")
    # Create model
    mlp = create_mlp(depth, width, activation_fn, alpha)

    with torch.no_grad():
        for p in mlp.parameters():
            p.data = initialization_scale * p.data

    # create optimizer
    assert optimizer_choice in optimizer_dict, f"Unsupported optimizer choice: {optimizer_choice}"
    optimizer = optimizer_dict[optimizer_choice](mlp.parameters(), lr=lr, weight_decay=weight_decay)

    # define loss function
    assert loss_function in loss_function_dict
    loss_fn = loss_function_dict[loss_function]()


    results[alpha] = {'train_loss': [], 'test_loss': [],
                        'train_accuracies': [], 'test_accuracies':[],
                        'log_steps':[],
                        'norms':[], 'last_layer_norms':[]}


    steps = 0
    one_hots = torch.eye(10, 10).to(device)

    with tqdm(total=optimization_steps) as pbar:
        for x, labels in islice(cycle(train_loader), optimization_steps):
            if (steps < 30) or (steps < 150 and steps % 10 == 0) or steps % log_freq == 0:
                results[alpha]['train_loss'].append(compute_loss(mlp, train, loss_function, device, N=len(train)))
                results[alpha]['train_accuracies'].append(compute_accuracy(mlp, train, device, N=len(train)))
                results[alpha]['test_loss'].append(compute_loss(mlp, test, loss_function, device, N=len(test)))
                results[alpha]['test_accuracies'].append(compute_accuracy(mlp, test, device, N=len(test)))
                results[alpha]['log_steps'].append(steps)
                with torch.no_grad():
                    total = sum(torch.pow(p, 2).sum() for p in mlp.parameters())
                    results[alpha]['norms'].append(float(np.sqrt(total.item())))
                    last_layer = sum(torch.pow(p, 2).sum() for p in mlp.mlp[-1].parameters())
                    results[alpha]['last_layer_norms'].append(float(np.sqrt(last_layer.item())))
                pbar.set_description("L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                    results[alpha]['train_loss'][-1],
                    results[alpha]['test_loss'][-1],
                    results[alpha]['train_accuracies'][-1] * 100,
                    results[alpha]['test_accuracies'][-1] * 100))
                
            if steps in [1, 10, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]:
                print(f"steps: {steps}; save model")
                save_path = os.path.join('/data/cici/Geometry/new/MNIST/checkpoints', f'alpha_{alpha}_{steps}.pth')
                torch.save(mlp.state_dict(), save_path)

            optimizer.zero_grad()
            y = mlp(x.to(device))
            if loss_function == 'CrossEntropy':
                loss = loss_fn(y, labels.to(device))
            elif loss_function == 'MSE':
                loss = loss_fn(y, one_hots[labels])
            loss.backward()
            optimizer.step()
            steps += 1
            pbar.update(1)


filename = os.path.join('/data/cici/Geometry/new/MNIST/checkpoints', f'acc_loss_log.pkl')
with open(filename, 'wb') as file:
    pkl.dump(results, file)

print(f"Results saved as {filename}")