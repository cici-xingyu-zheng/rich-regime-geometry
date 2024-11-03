
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


# Main analysis functions:
def kernel_distance(Kt1, Kt2):
    """
    Compute the kernel distance between two Neural Tangent Kernels (PyTorch tensor).
    """
    # make sure inputs are on the same device
    if Kt1.device != Kt2.device:
        Kt2 = Kt2.to(Kt1.device)

    frobenius_inner_product = torch.sum(Kt1 * Kt2) # Kt is symmetric
    
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
        
        if isinstance(batch, list):
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
        
        # Move outputs to CPU for further processing... 
        # probably cost us computational time; but safer to run
        outputs = outputs.cpu() 
        
        batch_jacobians = []
        for output in outputs:
            jacobian_rows = []
            # Get gradient for each fi: 
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