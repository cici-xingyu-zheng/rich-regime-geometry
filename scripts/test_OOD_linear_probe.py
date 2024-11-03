import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torchvision
from torch.utils.data import DataLoader, Subset
import os
import random
import numpy as np
from neuronet.utils import models

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

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

def evaluate_with_linear_probe(model, subsample_seed, num_batches=10):
    # local seed for subsampling
    rng = random.Random(subsample_seed)
    
    # Set up MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    
    num_samples = batch_size * num_batches
    
    indices = rng.sample(range(len(mnist_dataset)), num_samples)
    subset_dataset = Subset(mnist_dataset, indices)
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)
    
    # Extract features from the trained model
    features = []
    labels = []
    model.eval()
    
    def hook(module, input, output):
        features.append(output.cpu())
    
    # Register to the last layer
    model.layers[-1].register_forward_hook(hook)
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            _ = model(images)
            labels.append(targets)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    input_dim = features.shape[1]
    MNIST_num_classes = 10
    
    classifier = LinearClassifier(input_dim, MNIST_num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)
    
    # linear classifier
    num_epochs = 10
    
    for epoch in range(num_epochs):
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i+batch_size].to(device)
            batch_labels = labels[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = classifier(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate the linear classifier
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(features.to(device))
        _, predicted = torch.max(outputs, 1)
        accuracy = torch.sum(predicted == labels.to(device)).item() / len(labels)
    
    return accuracy

ood_accs_p5 = []

for run in range(10):
    print(f"\nStarting run {run + 1}/10")
    model_accs = []
    p5models = get_models(param_lists[0])
    
    for i, model in enumerate(p5models):
        print(f"Evaluating model checkpoint {i}")
        accuracy = evaluate_with_linear_probe(model, subsample_seed=run)
        print(f"Linear probing accuracy on MNIST: {accuracy:.4f}")
        model_accs.append(accuracy)
    
    ood_accs_p5.append(model_accs)

ood_accs_p5 = np.array(ood_accs_p5)

np.save('output/OOD/ood_accs_p5_include_different_seeds.npy', ood_accs_p5)

# Print final summary
ood_accs_mean = np.mean(ood_accs_p5, axis=0)
ood_accs_std = np.std(ood_accs_p5, axis=0)
print("\nFinal Results:")
for i, (mean, std) in enumerate(zip(ood_accs_mean, ood_accs_std)):
    print(f"Checkpoint {i}: Accuracy = {mean:.4f} ± {std:.4f}")