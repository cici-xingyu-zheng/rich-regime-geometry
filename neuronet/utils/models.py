import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


    
# MLP model
hidden_sizes = [128, 512, 128] # can sandwich a larger width? or increase them all to be large..
class MLP(nn.Module):
    def __init__(self, num_classes, input_size = 28*28, hidden_sizes = hidden_sizes, output_scale=1.0):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.output_scale = output_scale
        
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_classes))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  
        x = self.layers(x)
        return x * self.output_scale


# CNN model, didn't end up using
class CNN(nn.Module):
    def __init__(self, num_classes, output_scale=1.0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.output_scale = output_scale
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x * self.output_scale


def train(model, train_loader, test_loader, criterion, optimizer, device, num_epochs, checkpoint_dir=None):
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    progress_bar = tqdm(range(num_epochs))

    for epoch in progress_bar:
        # train:
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # forward pass
            outputs = model(images)
            if isinstance(criterion, nn.MSELoss):
                one_hot_labels = nn.functional.one_hot(labels, num_classes=model.num_classes).float()
                loss = criterion(outputs, one_hot_labels)
            else:
                loss = criterion(outputs, labels)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        train_loss = train_loss / total
        train_losses.append(train_loss)

        # eval:
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if isinstance(criterion, nn.MSELoss):
                    one_hot_labels = nn.functional.one_hot(labels, num_classes=model.num_classes).float()
                    loss = criterion(outputs, one_hot_labels)
                else:
                    loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        test_loss = test_loss / total
        test_losses.append(test_loss)

        # progress bar:
        progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

        # save model checkpoint //added 04/28/24
        # if checkpoint_dir and (epoch + 1) % 500 == 0:
        # if checkpoint_dir and (epoch + 1) in [5000, 6000, 7000, 8000, 9000, 10000]:
        # if checkpoint_dir and ((epoch + 1) % 500 == 0 or  (epoch + 1) in [5000, 6000, 7000, 8000, 9000, 10000]):
        # if checkpoint_dir and  (epoch + 1) in [50, 100, 200, 300, 400]:
        # to_save = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000]  # added 05/13/24    
        to_save = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000]  # added 05/13/24    

        if checkpoint_dir and  (epoch + 1) in to_save: 

            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            checkpoint = {  'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")

    return train_accuracies, test_accuracies, train_losses, test_losses

def visualize(train_accuracies, test_accuracies, train_losses, test_losses, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(train_accuracies, label='Train Accuracy')
    ax1.plot(test_accuracies, label='Test Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_xscale('log') 
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training and Test Accuracy')
    ax1.set_xlim([1,10000])
    ax1.legend()
    
    ax2.plot(train_losses, label='Train Loss')
    ax2.plot(test_losses, label='Test Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Log Loss')
    ax2.set_xscale('log') 
    ax2.set_yscale('log') 
    ax2.set_title('Training and Test Loss')
    ax2.set_xlim([1, 10000])
    ax2.legend()
    
    plt.suptitle(title)

    plt.tight_layout()
    plt.show()
    # Save the lists into a single .npy file
    np.savez(f'../output/{title}_training_data.npy', train_accuracies=train_accuracies, test_accuracies=test_accuracies,
            train_losses=train_losses, test_losses=test_losses)
    fig.savefig(f'../output/{title}.pdf')


