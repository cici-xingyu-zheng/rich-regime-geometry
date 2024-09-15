
import torch
import os
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
            # for each parameter tensor p, p.grad retrieves the gradient of that parameter
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
    # iterate over batches
    for _, (inputs, _) in enumerate(test_dataloader):

        # move the input data to the same device as the model
        inputs = inputs.to(next(model.parameters()).device)
    
        S = weight_jacobian_svd(model, inputs, normalized = normalized)

        # accumulate the sum of batch-wise Jacobian averages
        if S_sum is None:
            S_sum = S
        else:
            S_sum += S
        
        S_count += 1

    # compute the overall average Jacobian matrix
    S_average = S_sum / S_count

    return S_average.cpu().numpy()


def input_jacobian_average(model, data_loader, dim = (47, 784)):
    model.eval()
    
    jacobian_sum = None
    jacobian_count = 0
    
    # iterate over batches
    for _, (inputs, _) in enumerate(data_loader):
        # Move the input data to the same device as the model
        inputs = inputs.to(next(model.parameters()).device)

        batch_jacobian_sum = None
        batch_jacobian_count = 0
        
        for data_point in inputs:
            jacobian = torch.autograd.functional.jacobian(model, data_point, create_graph=False, strict=False)

            # reshape the Jacobian matrix to (output_dim, input_dim)
            jacobian = jacobian.view(dim[0], dim[1])  # Output dim: 47, Input dim: 784 (flattened 28x28)

            # accumulate the sum within the batch
            if batch_jacobian_sum is None:
                batch_jacobian_sum = jacobian
            else:
                batch_jacobian_sum += jacobian
            
            batch_jacobian_count += 1
        
        # compute the average Jacobian matrix within the batch
        batch_jacobian_average = batch_jacobian_sum / batch_jacobian_count
        
        # accumulate the sum of batch-wise averages
        if jacobian_sum is None:
            jacobian_sum = batch_jacobian_average
        else:
            jacobian_sum += batch_jacobian_average
        
        jacobian_count += 1
    
    # the overall average Jacobian matrix
    jacobian_average = jacobian_sum / jacobian_count
    
    return jacobian_average



def input_signular_average(jacobian_average, normalized = False):

    if normalized: 
        jacobian_norm = torch.norm(jacobian_average, p='fro')
        jacobian_average = jacobian_average / jacobian_norm
 
    _, S, _ = torch.linalg.svd(jacobian_average)

    return S.cpu().numpy()


def plot_spectrum(S0, Ss, S0_normalized, Ss_normalized, epochs, output_dir, wrt = 'weight'):
    '''
    Plot the singular value spectra
    '''
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
    plt.subplots_adjust(wspace=0.2) 
    filename = os.path.join(output_dir, f'J-{wrt}.pdf')
    fig.savefig(filename,  bbox_inches='tight')
    plt.close(fig)  



def compute_ntk_at_epoch(model, dataloader, device):
    '''
    Compute averaged NTK across batches.
    '''
    for param in model.parameters():
        param.requires_grad = True
    
    ntk_sum = None
    num_batches = 0

    # Concstruct the batch wise Jacobian matrix
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        jacobian = []
        
        for output in outputs:
            grad_output = torch.zeros_like(output)
            grad_output[:] = 1.0
            gradients = torch.autograd.grad(output, model.parameters(), grad_outputs=grad_output, create_graph=True)
            # flattening to be a row for the output
            jacobian.append(torch.cat([g.view(-1) for g in gradients]))
        
        # stack rows:
        jacobian = torch.stack(jacobian)
        
        # K(X, X'):
        ntk_batch = torch.matmul(jacobian, jacobian.t())
        
        # accumulate the NTK sum
        if ntk_sum is None:
            ntk_sum = ntk_batch.cpu().detach()
        else:
            ntk_sum += ntk_batch.cpu().detach()
        
        num_batches += 1
    
    # the average NTK over all batches
    ntk_avg = ntk_sum / num_batches
    
    return ntk_avg.numpy()

def kernel_distance(Kt1, Kt2):
    '''
    Compute the kernel distance between NTKs at different training times.
    '''
    # Frobenius inner:
    frobenius_inner_product = np.sum(Kt1 * Kt2)
    
    # Frobenius norms
    frobenius_norm_Kt1 = np.sqrt(np.sum(Kt1**2))
    frobenius_norm_Kt2 = np.sqrt(np.sum(Kt2**2))
    
    # the kernel distance
    kernel_dist = 1 - frobenius_inner_product / (frobenius_norm_Kt1 * frobenius_norm_Kt2)
    
    return kernel_dist

def compute_ntk_full_at_epoch(model, dataloader, device):
    '''
    Compute NTK for full dataset loaded.
    '''
    for param in model.parameters():
        param.requires_grad = True
    
    all_jacobians = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        jacobian = []
        
        for output in outputs:
            grad_output = torch.zeros_like(output)
            grad_output[:] = 1.0
            gradients = torch.autograd.grad(output, model.parameters(), grad_outputs=grad_output, create_graph=True)
            jacobian.append(torch.cat([g.view(-1) for g in gradients]))
        
        # stack Jacobians for this batch
        jacobian = torch.stack(jacobian)
        
        # stack across batch
        all_jacobians.append(jacobian)
        all_labels.append(labels)
    
    # concatenate Jacobians and labels from all batches
    full_jacobian = torch.cat(all_jacobians, dim=0)
    full_labels = torch.cat(all_labels, dim=0)
    
    # compute the NTK:
    ntk = torch.mm(full_jacobian, full_jacobian.t())
    
    # return NTK and labels as NumPy arrays 
    return ntk.cpu().detach().numpy(), full_labels.cpu().numpy()


def compute_cka(K, y):
    
    """
    Compute Centered Kernel Alignment (CKA), between NTK features and the given task y(X):
    CKA =  (y^T K_t y) / (||K_t||_F ||y||^2)

    Centering K and y by the centering matrix: https://en.wikipedia.org/wiki/Centering_matrix
    Alternatively can just subtract the mean.
    """

    # One-hot encode the labels
    num_classes = len(np.unique(y))
    y_onehot = np.eye(num_classes)[y]
    
    # Compute centering matrix
    n = K.shape[0]
    I = np.eye(n)
    H = I - np.ones((n, n)) / n 
    
    # Center the kernel matrix
    K_centered = H @ K @ H

    # Center the one-hot encoded labels
    y_centered = H @ y_onehot
    
    # Compute numerator of CKA
    numerator = np.trace(y_centered.T @ K_centered @ y_centered)
    
    # Compute denominator of CKA
    denominator = np.linalg.norm(K_centered, ord='fro') * np.linalg.norm(y_centered, ord='fro')
    
    # Compute CKA
    cka = numerator / denominator
    
    return cka