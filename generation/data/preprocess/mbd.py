from typing import Dict, List
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
import pickle

# TODO: Эту функцию в препроцесс паркетов
def encode_client_ids(train_clients: Dict, test_clients: Dict, config) -> List[Dict]:
    clients = np.hstack((
        train_clients, test_clients
    ))
    
    encoder = LabelEncoder()
    encoder.fit(clients)
    train_clients = encoder.transform(train_clients)
    test_clients = encoder.transform(test_clients)
    
    config['client_ids_encoder_save_path'] = f"{config['vae_ckpt_dir']}/client_ids_encoder.pickle"
    
    with open(config['client_ids_encoder_save_path'], 'wb') as file:
        pickle.dump(encoder, file)
    
    return torch.tensor(train_clients), torch.tensor(test_clients)
