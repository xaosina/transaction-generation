from .DiT import DiT
from .unet import Unet1D
from .base import LDMDenoiser

def load_ldm_denoiser(name: str, params: dict) -> LDMDenoiser:

    if name =='Unet1D':
        return Unet1D(
            latent_size=params['input_size'],
            generation_len=params['generation_len'],
            reference_len=params['reference_len'],
            history_embedding_dim=params['history_encoder_dim'],
            num_classes=1,
            dim_t=params['dim_t'],
            base_factor=params['base_factor'],
        )

    if name == 'DiT':
        return DiT(
            latent_size=params['input_size'],
            generation_len=params['generation_len'],
            reference_len=params['reference_len'],
            history_embedding_dim = params['history_encoder_dim'],
            num_classes = 1,
            hidden_size=params['hidden_size'],
            depth=params['depth'],
            num_heads=params['num_heads'],
            mlp_ratio=params['mlp_ratio'],
        )

    raise Exception(f'Unknown denoiser name {name}')
