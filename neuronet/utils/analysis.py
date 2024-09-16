
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


def kernel_distance(Kt1, Kt2):
    """
    Compute the kernel distance between two Neural Tangent Kernels (PyTorch tensor).
    """
    # make sure inputs are on the same device
    if Kt1.device != Kt2.device:
        Kt2 = Kt2.to(Kt1.device)

    frobenius_inner_product = torch.sum(Kt1 * Kt2) # K symmetric
    
    frobenius_norm_Kt1 = torch.sqrt(torch.sum(Kt1**2))
    frobenius_norm_Kt2 = torch.sqrt(torch.sum(Kt2**2))
    
    kernel_dist = 1 - frobenius_inner_product / (frobenius_norm_Kt1 * frobenius_norm_Kt2)
    
    return kernel_dist

def compute_ntk(model, dataloader, device):
    '''
    Compute the Neural Tangent Kernel (NTK) for the given model and dataset.
    Most computations are performed on CPU to reduce GPU memory usage.
    '''
    model.eval()  
    for param in model.parameters():
        param.requires_grad = True

    all_jacobians = []

    for batch_idx, batch in enumerate(dataloader):
        # print(f"Processing batch {batch_idx + 1}")
        
        # handle different input types
        if isinstance(batch, list):
            inputs = batch[0]  
        elif isinstance(batch, tuple): # this is not needed; only in test examples
            inputs = batch[0]
        else:
            inputs = batch
        
        # convert to tensor 
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        
        inputs = inputs.to(device)
        # print(f"Batch {batch_idx + 1}: Input shape = {inputs.shape}")
        
        model.zero_grad()
        outputs = model(inputs) # [batch_size, output_size]
        # print(f"Batch {batch_idx + 1}: Output shape = {outputs.shape}")
        
        # move outputs to CPU for further processing... probably cost us computational time
        outputs = outputs.cpu() 
        
        batch_jacobians = []
        for output in outputs:
            jacobian_rows = []
            # get gradient for each fi: 
            for out_idx in range(output.size(0)):
                grad_output = torch.zeros_like(output) # [output_size]
                grad_output[out_idx] = 1.0
                gradients = torch.autograd.grad(output.to(device), model.parameters(), grad_outputs=grad_output.to(device), retain_graph=True)
                jacobian_row = torch.cat([g.cpu().flatten() for g in gradients]) #  [num_params]
                jacobian_rows.append(jacobian_row)
            jacobian = torch.stack(jacobian_rows) # [output_size, num_params]
            batch_jacobians.append(jacobian)
        
        batch_jacobian = torch.cat(batch_jacobians, dim=0) # [batch_size * output_size, num_params]
        all_jacobians.append(batch_jacobian)
        
        # print(f"Batch {batch_idx + 1}: Processed {batch_jacobian.shape[0]} inputs, Jacobian shape = {batch_jacobian.shape}")
        
        # free up memory
        del batch_jacobians, jacobian_rows, gradients, grad_output, outputs
        torch.cuda.empty_cache()  # If using GPU, just in case

    # concatenate all Jacobians on CPU 
    full_jacobian = torch.cat(all_jacobians, dim=0) # [load_size * output_size, num_params]
    print(f"Full Jacobian shape: {full_jacobian.shape}")

    # compute the NTK on CPU
    ntk = torch.mm(full_jacobian, full_jacobian.t())
    print(f"NTK shape: {ntk.shape}")

    print("NTK computation completed.")
    return ntk