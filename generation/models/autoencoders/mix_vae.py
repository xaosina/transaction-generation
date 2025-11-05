import math
from copy import deepcopy
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from ebes.types import Seq

from generation.data import batch_tfs
from generation.data.data_types import GenBatch, LatentDataConfig, PredBatch
from generation.data.utils import create_instances_from_module
from generation.models.autoencoders.base import AEConfig, BaseAE

from .modules import Tokenizer, Transformer, Reconstructor
from .utils import get_features_after_transform

from .vae import VAE 
from .base import BaselineAE

class MIX_VAE(BaseAE):

    def __init__(self, data_conf: LatentDataConfig, model_config):
        super().__init__()

        self.vae = VAE(data_conf,model_config)
        self.ae = BaselineAE(data_conf,model_config)
        self.encoder = self.vae.encoder
        self.decoder = self.ae.decoder


