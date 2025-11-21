from copy import deepcopy
from dataclasses import replace

import torch
import torch.nn as nn
from ebes.model import BaseSeq2Seq
from ebes.types import Seq

from generation.models import autoencoders
from generation.utils import freeze_module

from ...data.data_types import GenBatch, LatentDataConfig, PredBatch, valid_mask
from ..encoders import AutoregressiveEncoder
from . import BaseGenerator, ModelConfig


class ConditionalHeadS(BaseSeq2Seq):
    """
    修改后的 Head：支持接收拼接了 Style 向量的上下文。
    """

    def __init__(self, context_size, style_size, k):
        super().__init__()
        self.context_size = context_size # D_gru
        self.style_size = style_size     # D_vae (z_style)
        
        # 拼接后的总输入维度
        self.total_input_dim = context_size + style_size 
        
        # 修改 projection 层以接受总维度
        # 输入: (Context + Style) + Queries(我们假设Queries维度与Context一致或独立定义)
        # 这里我们设定 Queries 维度等于 context_size
        self.proj = torch.nn.Linear(self.total_input_dim + context_size, context_size)
        self.relu = torch.nn.ReLU()

        # Queries 保持与 GRU 输出维度一致，或者是独立的维度
        self.queries = torch.nn.Parameter(torch.randn(k, context_size)) 
        self.k = k

    @property
    def output_dim(self):
        return self.context_size * self.k

    def forward_impl(self, ctx):
        # ctx shape: [B, D_gru + D_style]
        b, d = ctx.shape
        assert d == self.total_input_dim, f"Input dim mismatch: expected {self.total_input_dim}, got {d}"

        # 1. 准备 Queries: [B, K, D_gru]
        x = self.queries[None].repeat(b, 1, 1) 
        
        # 2. 准备 Context: [B, 1, D_gru + D_style] -> [B, K, D_gru + D_style]
        ctx_expanded = ctx.unsqueeze(1).repeat(1, self.k, 1)
        
        # 3. 拼接 Context 和 Queries
        # Result: [B, K, (D_gru + D_style) + D_gru]
        combined = torch.cat([ctx_expanded, x], -1)
        
        # 4. Flatten and Project
        combined = combined.flatten(0, 1) # (BK, D_total + D_query)
        out = self.proj(combined)         # (BK, D_out)
        out = self.relu(out)
        
        return out.reshape(b, self.output_dim) # (B, KO)

    def forward(self, seq: Seq):
        mask = valid_mask(seq)
        x = seq.tokens
        assert x.ndim > 2  # (L, B, D_total).
        shape = list(x.shape)
        x_masked = x[mask]  # (V, D_total).
        v = len(x_masked)
        
        # 处理 Masked 序列
        x_mapped = self.forward_impl(x_masked.flatten(0, -2)).reshape(
            *([v] + shape[2:-1] + [self.output_dim])
        ) 
        
        x_new = torch.zeros(
            *[shape[:-1] + [self.output_dim]],
            dtype=x_mapped.dtype,
            device=x_mapped.device
        )
        x_new[mask] = x_mapped
        return replace(seq, tokens=x_new)


class DeTPP_abs_style(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()

        # --- 1. 初始化 VAE (Frozen) ---
        self.autoencoder = getattr(autoencoders, model_config.autoencoder.name)(
            data_conf, model_config
        )
        self.autoencoder_name = model_config.autoencoder.name
        if model_config.autoencoder.checkpoint:
            ckpt = torch.load(model_config.autoencoder.checkpoint, map_location="cpu")
            self.autoencoder.load_state_dict(ckpt["model"], strict=False)
        
        # 强制冻结 VAE，不管配置文件怎么写，为了保证逻辑正确
        self.autoencoder = freeze_module(self.autoencoder)

        # 获取维度
        self.vae_dim = self.autoencoder.encoder.output_dim # D_vae (也是 Style 维度)
        
        # --- 2. 初始化 GRU (Encoder) ---
        encoder_params = model_config.latent_encoder.params or {}
        encoder_params["input_size"] = self.vae_dim 

        self.encoder = AutoregressiveEncoder(
            model_config.latent_encoder.name, encoder_params
        )
        # 获取 GRU 的输出维度
        self.gru_dim = self.encoder.output_dim

        # --- 3. 初始化 Head ---
        k_factor = model_config.params["k_factor"]
        assert k_factor >= 1
        self.k_output = int(k_factor * data_conf.generation_len)
        self.k_gen = model_config.params.get("k_gen") or data_conf.generation_len
        
        # 关键修改：传入 context_size 和 style_size
        self.next_k_head = ConditionalHeadS(
            context_size=self.gru_dim, 
            style_size=self.vae_dim, 
            k=self.k_output
        )
        
        # Presence Head 也需要能够处理拼接后的维度 (或者只处理 GRU 维度，看具体设计)
        # 为了简单和一致，建议 Presence Head 也接收拼接维度，或者只用 projection
        # 这里假设 Presence 只需要 GRU 上下文即可判断是否发生，若需要 Style 也可拼接
        # 下面代码修改为接收拼接后的维度
        self.presence_head = nn.Linear(self.gru_dim + self.vae_dim, self.k_output)

        self.gru_dim = self.encoder.output_dim
        self.vae_dim = self.autoencoder.encoder.output_dim

        # 🟢 1. 定义 GRU 输出的 LayerNorm
        # nn.LayerNorm(normalized_shape) 作用于 GRU 的特征维度 (D_gru)
        self.norm_gru = nn.LayerNorm(self.gru_dim)

        # 🟢 2. 定义 VAE 潜在向量的 LayerNorm
        # 作用于 VAE 的潜在维度 (D_vae)
        self.norm_vae = nn.LayerNorm(self.vae_dim)

    def _apply_delta(self, x: GenBatch):
        x = deepcopy(x)
        deltas = x.time
        deltas[1:,:] = deltas[1:,:] - deltas[:-1,:]
        deltas[0, :] = 0
        x.time = deltas
        return x

    def _sort_time_and_revert_delta(self, hist, pred):
        order = pred.time.argsort(dim=0)
        for attr in ["time", "num_features", "cat_features"]:
            tensor = getattr(pred, attr)
            if tensor is None: continue
            shaped_order = order.reshape(*(list(order.shape) + [1] * (tensor.ndim - order.ndim)))
            tensor = tensor.take_along_dim(shaped_order, dim=0)
            setattr(pred, attr, tensor)
        pred.time += hist.time[hist.lengths - 1, torch.arange(hist.shape[1])]
        return pred

    # 辅助函数：拼接 (Conditioning)
    def _condition_sequence(self, h_gru_seq: Seq, z_style: torch.Tensor) -> Seq:
        # h_gru_seq.tokens shape: [L, B, D_gru]
        # z_style shape:           [B, D_vae]
        
        L, B, _ = h_gru_seq.tokens.shape
        
        # 🟢 1. 归一化 GRU 序列
        # LayerNorm 自动作用于最后一个维度 (D_gru)，使得其均值为 0，方差为 1
        h_gru_norm = self.norm_gru(h_gru_seq.tokens) 
        
        # 🟢 2. 归一化 VAE 风格向量
        # LayerNorm 自动作用于最后一个维度 (D_vae)
        z_style_norm = self.norm_vae(z_style)        
        
        # 3. 扩展 VAE 风格向量
        z_repeated = z_style_norm.unsqueeze(0).repeat(L, 1, 1)
        
        # 4. 拼接
        conditioned_tokens = torch.cat([h_gru_norm, z_repeated], dim=-1)
        
        return replace(h_gru_seq, tokens=conditioned_tokens)


    def forward(self, x: GenBatch) -> PredBatch:
        L, B = x.shape
        x = deepcopy(x)
        if self.autoencoder_name == "BaselineAE":
            x = self._apply_delta(x)
            
        # 1. VAE Encoder -> Z_sequence [L, B, D_vae]
        # 由于 frozen=True 且内部 pretrained=False，这里返回的是包含采样 Z 的 Seq
        z_seq = self.autoencoder.encoder(x, copy=False) 
        # 2. GRU Encoder -> H_GRU [L, B, D_gru]
        h_gru_seq = self.encoder(z_seq) 
        
        # --- 3. 提取 Z_style 用于训练 (Reconstruction) ---
        # 取序列中最后一个有效时间步的 Z 作为 Style 的代表
        # z_seq.tokens: [L, B, D_vae]
        last_indices = z_seq.lengths - 1
        z_style_train = z_seq.tokens[last_indices, torch.arange(B)] # [B, D_vae]
        
        # 4. 拼接 (Conditioning)
        # 输入: GRU序列 + 提取的 Z_style
        x_conditioned = self._condition_sequence(h_gru_seq, z_style_train)
        
        # 5. Prediction
        # x_conditioned 的维度是 D_gru + D_vae，符合 Head 的要求
        x_pred = self.next_k_head(x_conditioned) # L, B, K * D
        
        x_pred = Seq(
            tokens=x_pred.tokens.reshape(L, B * self.k_output, -1),
            lengths=x_pred.lengths.repeat_interleave(self.k_output, 0),
            time=None,
        )
        
        # Presence Head 也使用拼接后的输入
        presence_scores = self.presence_head(x_conditioned.tokens).reshape(L, B, -1)
        
        # Decoder (Mapping back to features)
        x_recon = self.autoencoder.decoder(x_pred) 
        x_recon = x_recon.k_reshape(self.k_output) 

        return (x_recon, presence_scores,0.0)

    def generate(
        self,
        hist: GenBatch,
        gen_len: int,
        with_hist=False,
        topk=1,
        temperature=1.0,
    ) -> GenBatch:
        orig_hist = deepcopy(hist)
        hist = deepcopy(hist)
        already_generated = 0
        
        # --- 1. 采样全局风格 (实现多样性) ---
        # 在循环外采样一次，保持整个生成过程风格一致
        B = hist.shape[1]
        # 关键点：从 N(0, I) 采样，而不是从 VAE Encoder 获取
        z_style_prior = torch.randn(B, self.vae_dim, device=hist.device)
        
        with torch.no_grad():
            for _ in range(0, gen_len, self.k_gen):
                L, B = hist.shape
                x = deepcopy(hist)
                if self.autoencoder_name == "BaselineAE":
                    x = self._apply_delta(hist)
                
                # 2. VAE Encoder (仅用于提取历史特征)
                # 这里其实不需要 stochasticity，但因为它 frozen+pretrained=False，它会采样。
                # 这没关系，因为我们只用它来计算 GRU 的输入，而不用它来做 Style。
                z_seq_hist = self.autoencoder.encoder(x, copy=False)
                
                # 3. GRU Encoder -> H_GRU [1, B, D_gru] (AutoregressiveEncoder.generate 返回最后一步)
                h_gru_last = self.encoder.generate(z_seq_hist) 
                
                # 4. 拼接 (Conditioning)
                # 使用我们手动采样的 z_style_prior
                x_conditioned = self._condition_sequence(h_gru_last, z_style_prior)
                
                # 5. Prediction
                x_out = self.next_k_head(x_conditioned) 
                
                # Filter events logic ... (保持原样)
                # 注意 x_out.tokens shape 是 [1, B, K*D] -> reshape -> [K, B, D]
                x_tokens = x_out.tokens.reshape(B, self.k_output, -1).transpose(0, 1)
                
                # Presence Head 输入也要是拼接后的
                # x_conditioned.tokens [1, B, D_total] -> [B, D_total]
                p_in = x_conditioned.tokens.squeeze(0) 
                presence_scores = self.presence_head(p_in).transpose(0, 1) # [K, B]
                
                topk_indices = torch.topk(presence_scores, self.k_gen, dim=0)[1]
                x_tokens = torch.take_along_dim(x_tokens, topk_indices.unsqueeze(-1), dim=0)
                
                x_new_seq = Seq(
                    tokens=x_tokens,
                    lengths=torch.full((B,), self.k_gen, device=hist.device),
                    time=None,
                )
                
                # Reconstruct
                rec = self.autoencoder.decoder.generate(x_new_seq, topk=topk, temperature=temperature)
                
                already_generated += self.k_gen
                hist.append(rec)
        
        pred_batch = hist.tail(already_generated).head(gen_len)
        
        if with_hist:
            orig_hist.append(pred_batch)
            return orig_hist
        else:
            return pred_batch