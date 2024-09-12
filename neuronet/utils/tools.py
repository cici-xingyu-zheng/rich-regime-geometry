import torch
import os
import numpy as np

from neuronet.utils import models

num_classes = 47


def get_models(param_combo, checkpoint_parent_dir, output_parent_dir, epochs, device):

    subdir = '_'.join([f"{param_name}:{value}" for param_name, value in param_combo.items()])
    checkpoint_dir = os.path.join(checkpoint_parent_dir, subdir)
    output_dir = os.path.join(output_parent_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    # Load trained networks:
    MLPmodels = []
    for epoch in epochs:
        checkpoint = torch.load(f'{checkpoint_dir}/model_epoch_{epoch}.pth') ## check name
        model = models.MLP(num_classes, output_scale=param_combo['output_scale']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        MLPmodels.append(model)

    return MLPmodels, output_dir

def flatten_activations(activations):
    """
    Flatten the activations tensor by combining the non-batch dimensions.

    Args:
        activations (numpy.ndarray): The activations tensor.

    Returns:
        numpy.ndarray: The flattened activations tensor.
    """
    ndim = activations.ndim
    if ndim > 2:
        # Convolutional layer activations
        batch_size, channels, height, width = activations.shape
        flattened_size = channels * height * width
        flattened_activations = activations.reshape(batch_size, flattened_size)
    else:
        # Fully connected layer activations
        flattened_activations = activations

    return flattened_activations



def get_layer_activations(model, inputs, layer_name):
    # Initialize an empty list to store the layer activations
    activations = [] 

    def hook(module, input, output):
        # Define a hook function to capture the layer activations
        # This function will be called whenever the specified layer is activated during the forward pass
        activations.append(output.detach().cpu().numpy())  # Append the layer activations to the 'activations' list
    
    # Initialize a variable to store the hook handle
    handle = None 

    for name, module in model.named_modules():
        # Iterate over all the named modules in the model
        if name == layer_name:
            # If the current module name matches the specified layer name
            handle = module.register_forward_hook(hook)  # Register the hook function to the module
    
    # Perform a forward pass of the model on the given inputs
    model(inputs) 

    if handle is not None:
        # If a hook was registered (i.e., the specified layer was found)
        # Remove the hook to avoid any potential memory leaks
        handle.remove()  

    activations = np.concatenate(activations, axis=0)

    # added for conv layers
    flattened_activations = flatten_activations(activations)

    # Return the captured layer activations (assuming only one activation tensor is captured)
    return flattened_activations 

def reorganize_dict_to_list(dictionary):
    """
    Reorganizes a dictionary where the values are lists of activation tensors
    into a list of 2D NumPy arrays, where each array has the same dimensions,
    and the second dimension (sample count) is set to the smallest sample count
    among the dictionary items.

    Args:
        dictionary (dict): A dictionary where the keys are class names, and the
            values are lists of NumPy arrays representing sample activation tensors.

    Returns:
        list: A list of 2D NumPy arrays, where each array represents a class,
            and the dimensions are consistent across all arrays.
    """
    # Find the smallest sample count among the dictionary items
    min_sample_count = min(len(v) for v in dictionary.values())

    # Reorganize the dictionary into a list of 2D NumPy arrays
    reorganized_list = []
    for class_name, activations in dictionary.items():
        activation_array = np.stack(activations[:min_sample_count], axis=1)
        reorganized_list.append(activation_array)

    return reorganized_list


def get_model_activation(MLPmodels, activation_dir, test_dataloader, device, epochs):

    layer_names = [name for name, _ in MLPmodels[0].named_modules()]
    for i, model in enumerate(MLPmodels):
        print(f'model: {model}')
        for layer in layer_names[2:]:
            print(f'layer: {layer}')
            class_activations = {}
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Get the activations of the specified layer
                    activations = get_layer_activations(model, inputs, layer)  

                    for activation, label in zip(activations, labels):
                        if label.item() not in class_activations:
                             # Initialize an empty list for each class
                            class_activations[label.item()] = [] 
                        # Append the activation to the corresponding class list
                        class_activations[label.item()].append(activation)  
            activation_list = reorganize_dict_to_list(class_activations)

            ### replace with activation 
            np.save(f'{activation_dir}/epoch{epochs[i]}_{layer}.npy', activation_list)