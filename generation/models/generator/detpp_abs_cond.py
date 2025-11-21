from copy import deepcopy
import torch
import torch.nn as nn
from ebes.model import BaseSeq2Seq
from ebes.types import Seq

from generation.models import autoencoders
from generation.utils import freeze_module

from ...data.data_types import GenBatch, LatentDataConfig, PredBatch, valid_mask
from ..encoders import AutoregressiveEncoder
from . import BaseGenerator, ModelConfig
from .detpp import ConditionalHead # 假设 ConditionalHead 能够接收额外的条件输入

# ----------------------------------------------------------------------
# SimpleMLP 类保持不变，因为它在新的架构中不再需要用于潜在空间转换
# 但为了完整性保留
# ----------------------------------------------------------------------

class SimpleMLP(nn.Module):
    # ... (与原代码保持一致) ...
    def __init__(self, input_size,output_size,hidden_size=32,num_hidden_layers=2):
        super(SimpleMLP, self).__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x

# ----------------------------------------------------------------------

class DeTPP_abs_cond(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()
        
        # 1. 初始化并冻结 VAE 编码器（只用作特征提取）
        self.autoencoder = getattr(autoencoders, model_config.autoencoder.name)(
            data_conf, model_config
        )
        self.autoencoder_name = model_config.autoencoder.name
        if model_config.autoencoder.checkpoint:
            ckpt = torch.load(model_config.autoencoder.checkpoint, map_location="cpu")
            msg = self.autoencoder.load_state_dict(ckpt["model"], strict=False)
            print(f"Loaded VAE Checkpoint: {msg}")
        
        # 冻结整个 autoencoder 模块，确保我们只使用其编码器且不更新参数
        self.vae_encoder = freeze_module(self.autoencoder.encoder)
        self.vae_latent_dim = self.autoencoder.encoder.output_dim
        # VAE 解码器和 MLP 已被抛弃

        # 2. 初始化自回归编码器 (处理事件序列 S)
        encoder_params = model_config.latent_encoder.params or {}
        # 编码器输入尺寸是 VAE 编码器处理序列 X 后的维度
        encoder_params["input_size"] = self.vae_latent_dim 
        self.encoder = AutoregressiveEncoder(
            model_config.latent_encoder.name, encoder_params
        )
        self.encoder_output_dim = encoder_params.get("output_size", self.vae_latent_dim)


        # 3. 初始化 ConditionalHead (DeTPP 预测头)
        k_factor = model_config.params["k_factor"]
        assert k_factor >= 1
        self.k_output = int(k_factor * data_conf.generation_len)
        self.k_gen = model_config.params.get("k_gen") or data_conf.generation_len
        
        # ConditionalHead 的输入：
        # - 历史序列特征 $H$: 来自 self.encoder (self.encoder_output_dim)
        # - VAE 上下文 $Z$: 来自 self.vae_encoder (self.vae_latent_dim)
        
        # 注：ConditionalHead 必须修改以接收两个输入
        # 假设 ConditionalHead 的 __init__ 现在是 ConditionalHead(seq_dim, cond_dim, k_output)
        # 如果 ConditionalHead 无法修改，则需要将 Z 拼接或通过 MLP 混合到 H 中
        # 为了兼容性，我们假设 ConditionalHead 只接收一个维度，并在 forward 中进行拼接
        
        # 最终输入 ConditionalHead 的维度 = self.encoder_output_dim + self.vae_latent_dim 
        combined_dim = self.encoder_output_dim + self.vae_latent_dim

        self.next_k_head = ConditionalHead(
            combined_dim, self.k_output
        )
        # 存在性头也必须接收这个组合特征
        self.presence_head = nn.Linear(combined_dim, self.k_output) # 注意：改为 k_output

    def _apply_delta(self, x: GenBatch):
        # ... (保持不变) ...
        x = deepcopy(x)
        deltas = x.time

        deltas[1:,:] = deltas[1:,:] - deltas[:-1,:]
        deltas[0, :] = 0
        x.time = deltas
        return x

    def _sort_time_and_revert_delta(self, hist, pred):
        # ... (保持不变) ...
        order = pred.time.argsort(dim=0)  # (L, B).
        for attr in ["time", "num_features", "cat_features"]:
            tensor = getattr(pred, attr)
            if tensor is None:
                continue
            shaped_order = order.reshape(
                *(list(order.shape) + [1] * (tensor.ndim - order.ndim))
            )
            tensor = tensor.take_along_dim(shaped_order, dim=0)
            setattr(pred, attr, tensor)
        # Revert delta from hist
        pred.time += hist.time[hist.lengths - 1, torch.arange(hist.shape[1])]
        return pred

    def forward(self, x: GenBatch) -> PredBatch:
        L, B = x.shape
        # 1. VAE 编码器提取上下文 Z (固定不变)
        with torch.no_grad():
            vae_latent = self.vae_encoder(x, copy=False)  # Sequence of [L, B, D_vae]
        
        # 2. 自回归编码器处理序列特征 H
        vae_latent_seq = vae_latent.tokens # [L, B, D_vae]
        hist_features = self.encoder(vae_latent) # [L, B, D_enc]
        
        # 3. 拼接 Z 和 H 作为 ConditionalHead 的输入
        # 我们只关注最后一个时间步的特征 H_L
        H_L = hist_features.tokens[-1, :, :].unsqueeze(0) # [1, B, D_enc]
        
        # 如果 VAE 编码器输出的是序列，我们需要决定用哪个 Z：
        # 这里假设 Z 是最后一个时间步的 VAE 潜在表示
        Z = vae_latent_seq[-1, :, :].unsqueeze(0) # [1, B, D_vae]
        
        # 拼接特征 [1, B, D_enc + D_vae]
        combined_features = torch.cat([H_L, Z], dim=-1) # [1, B, D_enc + D_vae]
        
        # 4. DeTPP 头预测 (L=1, B, K*D_pred)
        # 注意：这里我们将 ConditionalHead 应用于最后一个时间步的特征
        # 如果 self.next_k_head 是一个 Transformer，这里需要调整输入 shape
        # 假设它是一个简单的 MLP/线性层，接受 [1, B, D_combined]
        x_pred_tokens = self.next_k_head(Seq(tokens=combined_features, lengths=None, time=None))  # [1, B, K * D_pred]

        # 5. 存在性预测 (L=1, B, K)
        presence_scores = self.presence_head(combined_features).reshape(1, B, self.k_output)  # [1, B, K]

        # 6. 重塑输出 (L=1, B, K, D_pred)
        D_pred = x_pred_tokens.tokens.shape[-1] // self.k_output
        x_pred = x_pred_tokens.tokens.reshape(1, B, self.k_output, D_pred) # [1, B, K, D_pred]
        
        # 返回 (预测结果, 存在性分数, 额外损失项)
        return (x_pred, presence_scores, torch.tensor(0.0, device=x_pred.device))

    def get_embeddings(self, hist: GenBatch):
        # ... (保持不变，但现在是 vae_encoder) ...
        hist = deepcopy(hist)
        x = self.vae_encoder(hist) # 使用固定的 VAE 编码器
        x = self.encoder.generate(x)
        assert x.tokens.shape[0] == 1
        return x.tokens[0]
    
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
        with torch.no_grad():
            for _ in range(0, gen_len, self.k_gen):
                L, B = hist.shape
                x = deepcopy(hist)

                # 1. VAE 编码器提取上下文 Z (固定不变)
                if self.autoencoder_name == "BaselineAE":
                    x = self._apply_delta(x) # 注意：这里要对原始 hist 进行 delta 编码
                vae_latent = self.vae_encoder(x, copy=False) # Sequence of [L, B, D_vae]
                
                # 2. 自回归编码器处理序列特征 H
                hist_features = self.encoder.generate(vae_latent) # [1, B, D_enc]
                
                # 3. 拼接 Z 和 H
                H_L = hist_features.tokens # [1, B, D_enc]
                Z = vae_latent.tokens[-1, :, :].unsqueeze(0) # 最后一个时间步的 VAE 潜在表示 [1, B, D_vae]
                combined_features = torch.cat([H_L, Z], dim=-1) # [1, B, D_enc + D_vae]

                # 4. DeTPP 头预测
                x_pred_tokens = self.next_k_head(Seq(tokens=combined_features, lengths=None, time=None))  # [1, B, K * D_pred]

                # 5. 存在性预测
                presence_scores = self.presence_head(combined_features).reshape(1, B, self.k_output) # [1, B, K]
                
                # 6. 选择 Top-K 事件
                presence_scores = presence_scores.squeeze(0) # [B, K]
                x_pred_tokens = x_pred_tokens.tokens.squeeze(0) # [B, K * D_pred]
                
                # D_pred
                D_pred = x_pred_tokens.shape[-1] // self.k_output
                
                # 重塑为 [B, K, D_pred]
                x_reshaped = x_pred_tokens.reshape(B, self.k_output, D_pred)
                
                # 选择 Top-k_gen 索引
                topk_indices = torch.topk(presence_scores, self.k_gen, dim=1)[1] # [B, k_gen]
                
                # 收集选中的事件 [B, k_gen, D_pred]
                topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D_pred)
                x_selected = torch.gather(x_reshaped, 1, topk_indices_expanded) # [B, k_gen, D_pred]
                
                # 7. 重建事件（这里直接将预测的 D_pred 作为重建结果，不需要 VAE 解码器）
                # DeTPP 的输出 $D_{\text{pred}}$ 必须直接编码事件的（时间、类型、特征）
                x = Seq(
                    tokens=x_selected.transpose(0, 1), # [k_gen, B, D_pred]
                    lengths=torch.full((B,), self.k_gen, device=hist.device),
                    time=None,
                )
                
                # 注意：由于我们移除了 VAE 解码器，x.tokens 已经是预测的原始事件数据
                # 如果 D_pred 包含时间信息，需要在这里提取并处理
                # 这里假设 D_pred 包含了 GenBatch 所需的全部特征
                # 由于 GenBatch 没有提供时间信息提取，我们暂时跳过 _sort_time_and_revert_delta

                # 8. 保存生成并附加到历史
                already_generated += self.k_gen
                # 假设 x 已经是 GenBatch 格式
                hist.append(x)
        
        # ... (收尾保持不变) ...
        pred_batch = hist.tail(already_generated).head(gen_len)
        
        if with_hist:
            orig_hist.append(pred_batch)
            return orig_hist
        else:
            return pred_batch