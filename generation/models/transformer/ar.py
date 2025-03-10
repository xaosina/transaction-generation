"""

Code was written by https://github.com/enhuiz

"""

import torch
from einops import rearrange
from torch import Tensor
from tqdm import trange
from torch import nn
from .base import Base
import gc

class AR(Base):
         
    @property
    def hidden_dim(self) -> bool:
        return self._hidden_dim

    @hidden_dim.setter
    def hidden_dim(self, value):
        self._hidden_dim = value

    @property
    def masked_training(self) -> bool:
        return self._masked_training

    @masked_training.setter
    def masked_training(self, value):
        self._masked_training = value
    
    @property
    def codes_dim(self) -> int:
        return self._codes_dim

    @codes_dim.setter
    def codes_dim(self, value):
        self._codes_dim = value
        
    @property
    def codes_number(self) -> int:
        return self._codes_number

    @codes_number.setter
    def codes_number(self, value):
        self._codes_number = value
    
    @property
    def output_mode(self) -> str:
        return self._output_mode

    @property
    def input_mode(self) -> str:
        return self._input_mode
    
    @output_mode.setter
    def output_mode(self, value):
        self._output_mode = value

    @input_mode.setter
    def input_mode(self, value):
        self._input_mode = value

    @property
    def cat_loss(self) -> bool:
        return True
    
    @property
    def on_codes(self) -> bool:
        return False
    
    @property
    def n_resp_levels(self):
        return 1
    
    @property
    def n_prom_levels(self):
        return 16

    @property
    def casual(self):
        return True

    @property
    def use_stop_token(self):
        return False

    @property
    def norm_type(self):
        return "ln"

    @property
    def resp_loss_only(self):
        return False

    def _prune(self, l: Tensor):
        indices = (l == self.stop_token).nonzero()
        if len(indices) == 0:
                return l
        if self.n_resp_levels > 1:
            return l[: indices.min(dim=0).values[0]]
        else:
            return l[: indices.min().item()]

    @staticmethod
    def _unsqueeze_list(x_list, axis=-1):
        return [x.unsqueeze(dim=axis) for x in x_list]

    def forward(
        self,
        text_list: Tensor,
        train_seq: Tensor,
        target: Tensor | None = None,
        max_steps: int = 1000,
        sampling_temperature: float = 1.0,
        rvqvae_decoder=None
    ):
        if target is not None:
            return super().forward(
                text_list,
                train_seq,
                target,
                quant_levels=None,
                shift_targ_list=False,
                return_all_resp=False,
            )
        else:
            return self._generate(
                text_list,
                train_seq,
                max_steps,
                sampling_temperature,
                rvqvae_decoder,
            )

    def _generate(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        max_steps: int,
        sampling_temperature: float,
        rvqvae_decoder = None
    ):

        device = text_list[0].device
        B = proms_list.shape[0]

        match self.output_mode:
            case "16x":
                response_list: list[Tensor] = [
                    torch.empty(0, self.codes_number, device=device).long()
                    for _ in text_list
                    ]
                assert rvqvae_decoder is not None
                self.__decode = rvqvae_decoder.to(device).decode_codes

            case "16x48":
                response_list: list[Tensor] = [
                    torch.empty(0, self.codes_number, self.hidden_dim, device=device).float()
                    for _ in text_list
                    ]     
                
            case "48":
                response_list: list[Tensor] = [
                    torch.empty(0, self.hidden_dim, device=device).float()
                    for _ in text_list
                    ]
        
        for _ in trange(max_steps) if max_steps > 10 else range(max_steps):
            save_r, _ = super().forward(
                text_list,
                proms_list,
                sampling_temperature=sampling_temperature,
            )
            
            proms_list = torch.cat([proms_list, 
                                    self.__decode(save_r).unsqueeze(1)], dim=1)

            for i, ri in enumerate(save_r.unbind(dim=0)):

                response_list[i] = torch.cat([response_list[i], ri.unsqueeze(0)], dim=0)

        pruned = [rsp for rsp in response_list]

        return pruned

    def __decode(self, data):
        return data
