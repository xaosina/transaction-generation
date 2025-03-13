from typing import Dict, Literal, Tuple, List
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
import pickle

def get_numpy_memmap(data_path, memmode=True):
    if memmode:
        return np.load(data_path, )
    else:
        return np.load(data_path, allow_pickle=True)

def get_latent_data_info(
        config,
        partition: Literal['train', 'test'] = 'train',
    ) -> Tuple[
        torch.Tensor, # tx latent embeddings
        Dict, # {client ids, abs times, age categories}
    ]:                          

    embedding_save_path = f"{config['vae_ckpt_dir']}/{partition}_z.npy"
    embedding_codes_save_path = f"{config['vae_ckpt_dir']}/{partition}_z_codes.npy"
    embedding_residual_qz_save_path = f"{config['vae_ckpt_dir']}/{partition}_zqs.npy"
    client_ids_save_path = f"{config['vae_ckpt_dir']}/orig_{partition}_client_ids.npy" 
    abs_times_save_path = f"{config['vae_ckpt_dir']}/orig_{partition}_abs_times.npy"

    match config["DATA_MODE"]:
        case "num2cat":
            return (
                dict(
                    latents = get_numpy_memmap(embedding_save_path),
                    latent_codes = get_numpy_memmap(embedding_codes_save_path),
                    latent_residual_qz = None,
                    client_ids = get_numpy_memmap(client_ids_save_path, memmode=False),
                    abs_times = get_numpy_memmap(abs_times_save_path),
                )
            ) 
        case "num2num":
            return (
                dict(
                    latents = get_numpy_memmap(embedding_save_path),
                    latent_codes = None,
                    latent_residual_qz = None,
                    client_ids = get_numpy_memmap(client_ids_save_path, memmode=False),
                    abs_times = get_numpy_memmap(abs_times_save_path),
                )
            )
        case "cat2cat":
            return (
                dict(
                    latents = get_numpy_memmap(embedding_codes_save_path),
                    latents_codes = None,
                    client_ids = get_numpy_memmap(client_ids_save_path, memmode=False),
                    abs_times = get_numpy_memmap(abs_times_save_path),
                )
            )
        case "num2mnum":
            return (
                dict(
                    latents = get_numpy_memmap(embedding_save_path),
                    latent_codes = None,
                    latent_residual_qz = get_numpy_memmap(embedding_residual_qz_save_path),
                    client_ids = get_numpy_memmap(client_ids_save_path, memmode=False),
                    abs_times = get_numpy_memmap(abs_times_save_path),
                )
            )
        case _:
            raise ValueError("No such mode of training")