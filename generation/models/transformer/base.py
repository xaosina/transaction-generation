"""

Code was written by https://github.com/enhuiz

"""

import math
from functools import partial
from typing import Literal, overload
from torch.profiler import record_function
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn
from torch.utils.checkpoint import checkpoint
from .adapter import CatOutputAdapter, Num16x48OutputAdapter, Num48OutputAdapter, NoSepInputAdapter


class SinusodialEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        exponent = torch.arange(self.d_half, dtype=torch.float32)
        exponent = exponent / self.d_half
        omega = torch.exp(-math.log(1e4) * exponent)
        self.omega: torch.Tensor
        self.register_buffer("omega", omega, persistent=False)

    @property
    def d_half(self):
        assert self.d_model % 2 == 0, "Only support even d_model."
        return self.d_model // 2

    def forward(self, x):
        """
        Args:
            x: (...)
        Returns:
            pe: (... d)
        """
        omega = self.omega

        while omega.dim() <= x.dim():
            omega = omega.unsqueeze(0)  # (... d)

        x = x.unsqueeze(-1)  # (... 1)
        x = omega * x
        x = torch.cat([x.sin(), x.cos()], dim=-1)

        return x

    def get_pe(self, n: int):
        """
        Args:
            n: int
        Returns:
            pe: (n d)
        """
        device = self.omega.device
        return self.forward(torch.arange(n, device=device))

    def add_pe(self, x):
        """
        Args:
            x: (b t c)
        """
        e = self.get_pe(x.shape[1])  # t d
        e = e[None]  # b t d
        x = x + e
        return x


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, casual):
        super().__init__()
        assert d_model % n_heads == 0
        dim_head = d_model // n_heads
        self.casual = casual
        self.n_heads = n_heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Linear(d_model, d_model)

    def forward(self, x, m):
        """
        Args:
            x: (b t c)
            m: (b t c), 1 is data, 0 is padding
        Returns:
            x: (b t c)
        """
        h = self.n_heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b t h d", h=h), (q, k, v))

        e = einsum("b i h d, b j h d -> b i j h", q, k)
        e = e * self.scale

        kpm = m.unsqueeze(1) * m.unsqueeze(2)  # b i j 1

        if self.casual:
            kpm = kpm.squeeze(-1).tril().unsqueeze(-1)  # b i j 1

        e = e.masked_fill(kpm == 0, -torch.finfo(e.dtype).max)
        a = e.softmax(dim=2)  # Normalize on j, i.e. key

        o = einsum("b i j h, b j h d -> b i h d", a, v)
        o = o.flatten(-2)
        o = self.to_out(o)  # b t c

        o = o * m

        return o


class AdaLN(nn.Module):
    def __init__(self, d_model, n_levels, eps=1e-5, k=0.1, c=2):
        super().__init__()
        self.eps = eps
        self.emb = nn.Embedding(n_levels, d_model * 2)
        self.k = k
        self.c = c
        nn.init.zeros_(self.emb.weight)

    def forward(self, x, l):
        logγ, β = self.emb(l).unsqueeze(1).chunk(2, dim=-1)

        h = F.layer_norm(x, x.shape[-1:], eps=self.eps)

        # The initial implementation (https://github.com/enhuiz/vall-e/blob/fbf023448c08e55c0422eefed7fc234cf8b76680/vall_e/vall_e/base.py#L135)
        # performed worse than vanilla LayerNorm.
        # The authors mentioned another AdaNorm paper (https://openreview.net/pdf?id=HyxndNrxLB) as they introduce AdaLN.
        # Did they use AdaNorm inside AdaLN? (as follows)
        h = self.c * (1 - (self.k * h).detach()) * h

        y = logγ.exp() * h + β

        return y


class PrenormResidual(nn.Module):
    def __init__(
        self,
        block,
        d_model,
        p_dropout,
        requires_mask=False,
        norm_type="ln",
        n_levels: int | None = None,
    ):
        super().__init__()
        self.block = block
        self.requires_mask = requires_mask
        self.norm_type = norm_type
        if norm_type == "ln":
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == "adaln":
            raise NotImplementedError('Check implementation')
            assert n_levels is not None
            self.norm = AdaLN(d_model, n_levels)
        else:
            raise NotImplementedError(norm_type)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, m, l):
        """
        Args:
            x: input (b t d)
            m: mask (b t 1), 1 is valuable and 0 is padding
            l: level to use, required only for AdaLN
        """
        nopts = {"l": l} if self.norm_type == "adaln" else {}
        bopts = {"m": m} if self.requires_mask else {}
        x = x + self.dropout(self.block(self.norm(x, **nopts) * m, **bopts))
        return x * m


class Block(nn.Sequential):
    def __init__(self, d_model, n_heads, p_dropout, casual, norm_type, n_levels):
        super().__init__()
        self.attn = PrenormResidual(
            Attention(d_model, n_heads, casual),
            d_model=d_model,
            p_dropout=p_dropout,
            requires_mask=True,
            norm_type=norm_type,
            n_levels=n_levels,
        )
        self.ffn = PrenormResidual(
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(p_dropout),
                nn.Linear(d_model * 4, d_model),
            ),
            d_model=d_model,
            p_dropout=p_dropout,
            norm_type=norm_type,
            n_levels=n_levels,
        )

    def forward(self, x, m, l):
        """
        Args:
            x: (b t c)
            m: (b t 1)
            l: (b)
        """
        poor_in_vram = True
        if x.requires_grad and poor_in_vram:
            x = checkpoint(self.attn, x, m, l, use_reentrant=False)
        else:
            x = self.attn(x, m, l)
        x = self.ffn(x, m, l)
        return x


class Embedding(nn.Embedding):
    def forward(self, x_list: list[Tensor]) -> list[Tensor]:
        if len(x_list) == 0:
            return []
        if isinstance(x_list, list):
            return super().forward(torch.cat(x_list)).split([*map(len, x_list)])
        else:
            a = super().forward(torch.cat([x for x in x_list]))
            b = a.split([*map(len, x_list)])
            return torch.stack(b)


class MultiEmbedding(nn.Module):
    """
    This embedding sums embeddings on different levels.
    """

    def __init__(self, max_n_levels, n_tokens, token_dim):
        super().__init__()
        self.max_n_levels = max_n_levels
        self.n_tokens = n_tokens
        self.weight = nn.Parameter(torch.randn(max_n_levels, n_tokens, token_dim))

    def forward(self, x_list: list[Tensor]) -> list[Tensor]:
        if len(x_list) == 0:
            return []

        w = self.weight

        padded_x_list = []

        for xi in x_list:
            xi = F.one_hot(xi, num_classes=self.n_tokens)  # t l' k
            assert w.shape[0] - xi.shape[1] >= 0, (w.shape[0], xi.shape[1])
            xi = F.pad(xi, (0, 0, 0, w.shape[0] - xi.shape[1]))  # t l k
            padded_x_list.append(xi.to(w))

        x = torch.cat(padded_x_list)  # n l k
        x = einsum("l k d, n l k -> n d", w, x)

        x_list = x.split([*map(len, x_list)])

        return x_list


class DirectLinearEmbedding(nn.Module):

    def __init__(self, d_latent, d_model):
        super().__init__()
        self.proj = nn.Linear(d_latent, d_model)

    def forward(self, x_list: torch.Tensor) -> list[torch.Tensor]:
        if isinstance(x_list, list):
            lengths = [x.shape[0] for x in x_list]

            x_cat = torch.cat(x_list, dim=0)

            out_cat = self.proj(x_cat)  # [sum(T_i), d_model]

            out_list = torch.split(out_cat, lengths, dim=0)
            
            return list(out_list)
        else:
            out_cat = self.proj(x_list)  
            return out_cat


class Masker(nn.Module):

    def __init__(self, 
                 B: int, 
                 L: int,
                 max_zero_prob=0.8,
                 n_masks=500,
                 ):
        super().__init__()
        
        self.B = B
        self.L = L
        self.n_masks = n_masks
        
        zero_probs = torch.linspace(0.0, max_zero_prob, steps=n_masks)
        
        mask_bank = []
        for p in zero_probs:
            mask = torch.bernoulli(torch.full((B, L), 1 - p))
            mask_bank.append(mask)

        mask_bank = torch.stack(mask_bank, dim=0)
        
        self.register_buffer('mask_bank', mask_bank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        i = torch.randint(0, self.n_masks, (1,), device=x.device)

        mask = self.mask_bank[i]
        mask = mask.squeeze(0)
        
        row_perm = torch.randperm(x.shape[0], device=x.device)
        col_perm = torch.randperm(x.shape[1], device=x.device)
        
        mask = mask[row_perm][:, col_perm]
        mask = mask.unsqueeze(-1)
        
        return x * mask

class Base(nn.Module):

    @property
    def casual(self) -> bool:
        raise NotImplementedError

    @property
    def n_resp_levels(self) -> int:
        raise NotImplementedError

    @property
    def use_stop_token(self) -> bool:
        raise NotImplementedError

    @property
    def norm_type(self):
        raise NotImplementedError
    
    @property
    def resp_loss_only(self):
        raise NotImplementedError

    def __init__(
        self,
        n_tokens: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        p_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self._input_mode = "none"
        self._codes_dim = -1
        self._codes_number = -1
        self._output_mode = "none"
        self._masked_training = False
        self._hidden_dim = -1
        casual = self.casual
        self.d_model = d_model

        blocks = [
            Block(
                d_model=d_model,
                n_heads=n_heads,
                p_dropout=p_dropout,
                casual=casual,
                norm_type=self.norm_type,
                n_levels=self.n_resp_levels, # TODO
            )
            for _ in range(n_layers)
        ]

        self.blocks = nn.ModuleList(blocks)


    # def init(self, loss_type, mask_params=None):
    #     assert self._hidden_dim != -1
    #     match self.input_mode:
    #         case "48":
    #             self.linear_proj = DirectLinearEmbedding(self.hidden_dim, self.d_model)
    #             self.last_linear = nn.Linear(self.d_model, self.hidden_dim)
    #             self.input_adapter = NoSepInputAdapter()
                
    #     match self.output_mode:
    #         case "16x":
    #             self.output_adapter = CatOutputAdapter(self.codes_dim, self.codes_number, loss_type=loss_type)
    #             self.last_linear = nn.Linear(self.d_model, self.d_model)
    #             self.classifier = nn.Linear(self.d_model, self.codes_dim * self.codes_number)
    #             # self.classifier = nn.ModuleList([nn.Linear(48, self.codes_dim) for _ in range(self.codes_number)])
    #         case "48":
    #             self.output_adapter = Num48OutputAdapter(loss_type=loss_type)
    #             self.last_linear = nn.Linear(self.d_model, self.d_model)
    #             self.classifier = nn.Linear(self.d_model, self.hidden_dim)
    #         case "16x48":
    #             self.output_adapter = Num16x48OutputAdapter(loss_type=loss_type)
    #             self.last_linear = nn.Linear(self.d_model, self.hidden_dim)
    #             self.classifier = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.codes_number)])

    #     if self.masked_training:
    #         self.masker = Masker(*mask_params)


    @property
    def stop_token(self):
        if not self.use_stop_token:
            raise ValueError("Not using stop token!")
        return self.n_tokens

    @property
    def ignore_index(self):
        return -100

    @staticmethod
    def _samplewise_merge_tensors(*l, sep: Tensor | None):
        if sep is None:
            return torch.cat(l, dim=1)
        else:
            batch_size = l[0].size(0)

            sep_shape = [batch_size, 1] + [-1] * len(sep.shape)
            sep_token = sep.unsqueeze(0).unsqueeze(1).expand(sep_shape)

            pieces = []

            for i, t in enumerate(l):
                pieces.append(t)
                if i < len(l) - 1:
                    pieces.append(sep_token)
            return torch.cat(pieces, dim=1)

    def forward(
            self,
            x: torch.Tensor,
            m: torch.Tensor,
        ):
            # if self.masked_training and target is not None:
            #     train_seq = self.masker(train_seq)

            # with record_function("Go"):
            #     x_list = self.input_adapter.prepare(self, text_list, train_seq)

            for block in self.blocks:
                x = block(x, m)
            
            return x

            x = self.output_adapter.after_attention(self, x, m)

            with record_function("compute logits"):
                h_list = self.output_adapter.compute_logits(self, x, m)

            with record_function("compute loss"):
                if target is not None:
                    self.loss = self.output_adapter.compute_loss(h_list, target, self.ignore_index)

            with record_function("Sample"):
                ret = self.output_adapter.sample(h_list, return_all_resp, sampling_temperature)

            return ret, None
