from neuronet.utils import models, analysis, tools

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
import pickle as pkl

num_classes = 47
batch_size = 128

# device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Set the random seed for Python and NumPy, just in case
random.seed(seed)
np.random.seed(seed)

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

# the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# paths relative to the script's location
output_parent_dir = os.path.join(script_dir, "..", "output/jacobian_SVD")

epochs = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000] 
# epochs = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000] 



# standard transform for EMNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

test_dataset = torchvision.datasets.EMNIST(root='./data', split='balanced', train=False, download=False, transform=transform)

subset_indices = np.random.choice(len(test_dataset), 1280, replace=False)
test_subset = torch.utils.data.Subset(test_dataset, subset_indices)
test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)


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

    #  ---------------- WEIGHT JACOBIAN -------------------------------
    print('Weight Jacobian!\n')

    S0 = analysis.weight_signular_average(model0, test_dataloader)

    Ss = []
    for i, model in enumerate(MLPmodels):
        print("comopute Jacobian", epochs[i])
        S = analysis.weight_signular_average(model, test_dataloader)
        Ss.append(S)

    S0_normalized = analysis.weight_signular_average(model0, test_dataloader, normalized= True)

    Ss_normalized = []
    for i, model in enumerate(MLPmodels):
        print("comopute Jacobian", epochs[i])
        S = analysis.weight_signular_average(model, test_dataloader, normalized= True)
        Ss_normalized.append(S)

    # plotting and saving
    analysis.plot_spectrum(S0, Ss, S0_normalized, Ss_normalized, epochs, output_dir, wrt = 'weight')

    data = {
        'S0': S0,
        'Ss': Ss,
        'S0_normalized': S0_normalized,
        'Ss_normalized': Ss_normalized
    }

    filename = os.path.join(output_dir, f'J-weight.pkl')
    with open(filename, 'wb') as file:
        pkl.dump(data, file)

    print(f"Data saved as {filename}")


    # ---------------- INPUT JACOBIAN -------------------------------
    print()
    print('Input Jacobian!\n')

    J_avg = analysis.input_jacobian_average(model0, test_dataloader)
    S0 = analysis.input_signular_average(J_avg)
    S0_normalized =  analysis.input_signular_average(J_avg, normalized= True)

    Ss = []
    Ss_normalized = []
    for i, model in enumerate(MLPmodels):
        print("comopute Jacobian", epochs[i])
        J_avg = analysis.input_jacobian_average(model, test_dataloader)
        S =  analysis.input_signular_average(J_avg)
        Ss.append(S)
        S =  analysis.input_signular_average(J_avg, normalized= True)
        Ss_normalized.append(S)

    # plotting and saving
    analysis.plot_spectrum(S0, Ss, S0_normalized, Ss_normalized, epochs, output_dir, wrt = 'input')

    data = {
        'S0': S0,
        'Ss': Ss,
        'S0_normalized': S0_normalized,
        'Ss_normalized': Ss_normalized
    }
    
    filename = os.path.join(output_dir, f'J-input.pkl')
    with open(filename, 'wb') as file:
        pkl.dump(data, file)

    print(f"Data saved as {filename}")