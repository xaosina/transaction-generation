import torch
import math

def sample_t(num_samples):
    return torch.rand(num_samples)

def get_loss_func(loss_type: str):
    if loss_type == 'l2-squared':
        def loss_func(x, y, A_prime):
            return torch.mean(A_prime**2 * (x - y)**2)
    
    elif loss_type == 'l2':
        def loss_func(x, y, A_prime):
            return torch.sqrt(torch.mean(A_prime**2 * (x - y)**2))
    
    elif loss_type == 'huber':
        def loss_func(x, y, A_prime):
            data_dim = x.shape[1] * x.shape[2] * x.shape[3]
            huber_c = 0.00054 * math.sqrt(data_dim)
            loss = torch.sum(A_prime**2 * (x - y)**2)
            loss = torch.sqrt(loss + huber_c**2) - huber_c
            return loss / data_dim
    
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented")
    
    return loss_func