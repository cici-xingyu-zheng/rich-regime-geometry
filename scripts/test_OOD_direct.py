import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
import numpy as np
import os 
from neuronet.utils import models
import random 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Set the random seed for Python and NumPy, just in case
random.seed(seed)
np.random.seed(seed)

param_names = ['optimizer', 'weight_scale', 'train_size', 'output_scale', 'weight_decay']

param_lists = [ 
                [ 'AdamW', 10, 2000, .5, 0],
                [ 'AdamW', 10, 2000, .001, 0],
            ]

checkpoint_parent_dir = '/data/cici/Geometry/new/checkpoints'
epochs = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000] 

EMNIST_num_classes = 47
batch_size = 128

def get_models(param_list):
    param_combo = dict(zip(param_names, param_list))
    subdir = '_'.join([f"{param_name}:{value}" for param_name, value in param_combo.items()])
    checkpoint_dir = os.path.join(checkpoint_parent_dir, subdir)
    # Load trained networks:
    MLPmodels = []
    for epoch in epochs:
        checkpoint = torch.load(f'{checkpoint_dir}/model_epoch_{epoch}.pth') ## check name
        model = models.MLP(EMNIST_num_classes, output_scale=param_combo['output_scale']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        MLPmodels.append(model)

    return MLPmodels


def evaluate_mnist_direct(models, device, num_samples=1280, batch_size=128, seed=42):
    """
    Directly evaluate MNIST test set using EMNIST model's original readout layer.
    
    Args:
        models: List of EMNIST models to evaluate
        device: torch device to use
        num_samples: Number of MNIST samples to evaluate on
        batch_size: Batch size for evaluation
        seed: Random seed for sample selection
    """
    # Random seed for each sampling
    random.seed(seed)
    torch.manual_seed(seed)
    
    # same transform as EMNIST:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    
    total_available = len(mnist_dataset)
    if num_samples > total_available:
        print(f"Warning: Requested {num_samples} samples but only {total_available} available.")
        num_samples = total_available
    
    indices = random.sample(range(total_available), num_samples)
    subset_dataset = Subset(mnist_dataset, indices)
    
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)
    
    num_batches = len(dataloader)
    print(f"Evaluating on {num_samples} samples ({num_batches} batches)")
    
    model_accs = []
    
    for i, model in enumerate(models):
        print(f"Evaluating model checkpoint {i}")
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(device)
                outputs = model(images)
                
                # Only consider first 10 logits (corresponding to digits 0-9)
                # outputs = outputs[:, :10]
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted.cpu() == targets).sum().item()
        
        accuracy = correct / total
        print(f"Checkpoint {i} - Direct evaluation accuracy on MNIST ({total} samples): {accuracy:.4f}")
        model_accs.append(accuracy)
    
    return model_accs

# evaluation multiple times
ood_accs_direct = []
num_samples = 1280 # (10 batches of 128)
num_runs = 10

for run in range(num_runs):
    print(f"\nStarting run {run + 1}/{num_runs}")
    p5models = get_models(param_lists[0])
    model_accs = evaluate_mnist_direct(
        models=p5models,
        device=device,
        num_samples=num_samples,
        batch_size=128,
        seed=run  # set differnet seed for each run
    )
    ood_accs_direct.append(model_accs)

ood_accs_direct = np.array(ood_accs_direct)

np.save(f'output/OOD/ood_accs_direct_{num_samples}_samples_all_classes.npy', ood_accs_direct)

# print summary
ood_accs_mean = np.mean(ood_accs_direct, axis=0)
ood_accs_std = np.std(ood_accs_direct, axis=0)
print("\nFinal Results:")
for i, (mean, std) in enumerate(zip(ood_accs_mean, ood_accs_std)):
    print(f"Checkpoint {i}: Accuracy = {mean:.4f} ± {std:.4f}")