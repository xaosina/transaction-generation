import math
import torch
import torch.nn as nn

from .DiT_models import DiT
from ebes.types import Seq
from ebes.model.basemodel import BaseModel
from torchdiffeq import odeint
from ....data.data_types import GenBatch, LatentDataConfig

class Reshaper(nn.Module):
    def __init__(self, gen_len: int):
        super().__init__()
        self.gen_len = gen_len

    def forward(self, seq: Seq) -> Seq:
        tensor = seq.tokens
        assert (
            tensor.shape[1] % self.gen_len == 0
        ), f"hidden_size doesnt divide by {self.gen_len}"
        B, D = tensor.shape
        return Seq(
            tokens=tensor.view(B, self.gen_len, D // self.gen_len).permute(1, 0, 2).contiguous(),
            lengths=torch.ones((B,), dtype=torch.long, device=tensor.device) * self.gen_len,
            time=None,
        )

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

class AsynDiffEncoder(BaseModel):

    def __init__(self, name: str, params: dict = None):

        super().__init__()

        self.params = params
        self.model = DiT(
        num_rows=params['num_rows'],
        latent_size=params['latent_size'],
        hidden_size=params['hidden_size'],
        depth=params['depth'],
        num_heads=params['num_heads'],
        mlp_ratio=params['mlp_ratio'],
        learn_sigma=params['learn_sigma']
    )
        self.loss_func = get_loss_func(params['loss_type'])
        self.gen_reshaper = Reshaper(params['generation_len'])
        self.mask = params['mask']
        self.generation_len = params['generation_len']
        
    def forward(self,target_seq: GenBatch,A):
        z = target_seq.tokens
        z = z.permute(1, 0, 2)
        attn_mask = A.attn_mask

        noise_fixed = torch.randn_like(z,device="cuda")

        # Sample t, zt
        t = sample_t(z.shape[0]).view(-1,1) # (batchsize,)
        A_t = A(t).to("cuda")
        A_t_dot = A.derivative(t).unsqueeze(-1).to("cuda")
        zt = A_t.unsqueeze(-1)*z+(1-A_t.unsqueeze(-1))*noise_fixed
        target = z - noise_fixed
        
        # Forward pass
        if self.mask:
            pred = self.model(zt, A_t, attn_mask)
        else:
            pred = self.model(zt, A_t)

        # Compute loss
        loss = self.loss_func(pred, target, A_t_dot)
        loss = loss.mean()
        return loss 

    def generate(self,z,gen_len,A,batch_len) -> GenBatch:

        # Create a mask
        z_tokens = z.tokens.permute(1, 0, 2)
        col_indices = torch.arange(z_tokens.shape[1]).unsqueeze(0).to("cuda")
        mask = col_indices < (batch_len - gen_len).unsqueeze(1)

        # Initiate noise
        noise_fixed = torch.rand_like(z_tokens)

        # Define the ODE function for solving the reverse flow
        def ode_func(t, x):
            t = t.view(-1,1)
            A_t = A(t)
            A_t_dot = A.derivative(t).unsqueeze(-1)
            # Compute vector field: x_0 - epsilon
            v = self.model(x,A_t)
            # Fix vector fields for preceding events
            v[mask] = (z_tokens-noise_fixed)[mask]
            return A_t_dot*v

        # Sample t, zt
        solution = odeint(ode_func, noise_fixed, A.times, rtol=1e-5, atol=1e-5, method=self.params["int_ode"])
        # Extract the result at t=0
        
        x_restored = solution[-1]

        pred = x_restored[:,self.generation_len:,:].view(z_tokens.shape[0],-1)

        return self.gen_reshaper(
            Seq(
                tokens=pred, 
                lengths=torch.ones((z_tokens.shape[0],)).to("cuda") * self.generation_len, 
                time=None
            )
        )


