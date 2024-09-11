
import torch
import numpy as np
import matplotlib.pyplot as plt



def weight_jacobian_svd(model, inputs, normalized = False):
    """
    Computes the SVD of the weight Jacobian for a given model and input batch.
    
    Args:
        model (torch.nn.Module): The model for which to compute the weight Jacobian SVD.
        x (torch.Tensor): The input batch tensor with shape (batch_size, input_dim).
        
    Returns:
        S (torch.Tensor): The singular values with shape (output_dim,).
    """
    output = model(inputs)
    output_dim = output.size(1)
    num_params = sum(p.numel() for p in model.parameters())
    
    def compute_jacobian():
        """
        Computes the weight Jacobian matrix.
        
        Returns:
            torch.Tensor: The weight Jacobian matrix with shape (num_params, output_dim).
        """
        jacobian = torch.zeros(num_params, output_dim, device=inputs.device)

        #iterates over each output dimension i and computes the Jacobian column for that output dimension 
        for i in range(output_dim):
            model.zero_grad()
            output[:, i].backward(torch.ones_like(output[:, i]), retain_graph=True) # contain the sum of gradients for all samples in the batch
            # For each parameter tensor p, p.grad retrieves the gradient of that parameter
            jacobian_i = torch.cat([p.grad.view(-1) for p in model.parameters()])
            jacobian[:, i] = jacobian_i
        
        return jacobian
    
    jacobian = compute_jacobian()

    if normalized:

        jacobian_norm = torch.norm(jacobian, p='fro')
        jacobian = jacobian / jacobian_norm
    
    #Compute the SVD of the weight Jacobian
    _, S, _ = torch.svd(jacobian) 

    return S

def weight_signular_average(model, test_dataloader, normalized = False):

    model.eval()
    S_sum = None
    S_count = 0
    # Iterate over batches
    for _, (inputs, _) in enumerate(test_dataloader):

        # Move the input data to the same device as the model
        inputs = inputs.to(next(model.parameters()).device)
    
        S = weight_jacobian_svd(model, inputs, normalized = normalized)

        # Accumulate the sum of batch-wise Jacobian averages
        if S_sum is None:
            S_sum = S
        else:
            S_sum += S
        
        # Increment the count of batches
        S_count += 1

    # Compute the overall average Jacobian matrix
    S_average = S_sum / S_count

    return S_average.cpu().numpy()


def input_jacobian_average(model, data_loader, dim = (47, 784)):
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize variables to store the sum and count of Jacobian matrices
    jacobian_sum = None
    jacobian_count = 0
    
    # Iterate over batches
    for _, (inputs, _) in enumerate(data_loader):
        # Move the input data to the same device as the model
        inputs = inputs.to(next(model.parameters()).device)

        # Initialize variables to store the sum and count of Jacobian matrices within the batch
        batch_jacobian_sum = None
        batch_jacobian_count = 0
        
        # Iterate over each data point in the batch
        for data_point in inputs:
            jacobian = torch.autograd.functional.jacobian(model, data_point, create_graph=False, strict=False)

            # Reshape the Jacobian matrix to (output_dim, input_dim)
            jacobian = jacobian.view(dim[0], dim[1])  # Output dim: 47, Input dim: 784 (flattened 28x28)

            # Accumulate the sum of Jacobian matrices within the batch
            if batch_jacobian_sum is None:
                batch_jacobian_sum = jacobian
            else:
                batch_jacobian_sum += jacobian
            
            # Increment the count of Jacobian matrices within the batch
            batch_jacobian_count += 1
        
        # Compute the average Jacobian matrix within the batch
        batch_jacobian_average = batch_jacobian_sum / batch_jacobian_count
        
        # Accumulate the sum of batch-wise Jacobian averages
        if jacobian_sum is None:
            jacobian_sum = batch_jacobian_average
        else:
            jacobian_sum += batch_jacobian_average
        
        # Increment the count of batches
        jacobian_count += 1
    
    # Compute the overall average Jacobian matrix
    jacobian_average = jacobian_sum / jacobian_count
    
    return jacobian_average



def input_signular_average(jacobian_average, normalized = False):

    if normalized: 
        jacobian_norm = torch.norm(jacobian_average, p='fro')
        jacobian_average = jacobian_average / jacobian_norm
 
    _, S, _ = torch.linalg.svd(jacobian_average)

    return S.cpu().numpy()


def plot_spectrum(S0, Ss, S0_normalized, Ss_normalized, epochs, name, wrt = 'weight'):
    # Plot the singular value spectra
    cmap = plt.get_cmap('plasma')
    norm = plt.Normalize(np.log10(min(epochs)), np.log10(max(epochs)))
    colors = [cmap(norm(np.log10(epoch))) for epoch in epochs]

    fig, ax = plt.subplots(ncols = 2, figsize=(12,5))

    ax[0].plot(S0, color = 'black', label = 'untrained')

    for i, (S, color )in enumerate(zip(Ss, colors)):
        ax[0].plot(S, color = color, label = f'epoch {epochs[i]}')
    ax[0].set_xlabel('Singular Value Index')
    ax[0].set_ylabel('Singular Value')
    ax[0].set_title(r'Singular Value Spectra [($J = \nabla_{\theta} f(x; \theta)$)]')


    ax[1].plot(S0_normalized, color = 'black', label = 'untrained')

    for i, (S, color )in enumerate(zip(Ss_normalized, colors)):
        ax[1].plot(S, color = color, label = f'epoch {epochs[i]}')
    ax[1].set_xlabel('Singular Value Index')
    ax[1].set_ylabel('Singular Value')
    ax[1].set_title(r'Normalized [ $J = \nabla_{\theta} f(x; \theta)/ ||\; J||_F$)]')
    ax[1].legend(loc = [1.01, -.05])
    plt.subplots_adjust(wspace=0.2)  # Adjust the spacing between subplots

    plt.show()

    fig.savefig(f'output/analysis/{name}_J-{wrt}.pdf',  bbox_inches='tight')