class UniModMLP(nn.Module):
    """
        Input: 序列数据，shape 为 (L, B, C)
            x_num: [L, B, d_numerical]
            x_cat: [L, B, len(categories)]
        Output:
            x_num_pred: [L, B, d_numerical]
            x_cat_pred: [L, B, sum(categories)]
    """
    def __init__(
            self, d_numerical, categories, num_layers, d_token,
            n_head = 1, factor = 4, bias = True, dim_t=512, use_mlp=True, **kwargs
        ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories

        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias = bias)
        self.encoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        
        # 计算 Transformer 编码器输出的序列长度 L_token_out
        # L_token_out = d_numerical + len(categories) (因为忽略了 CLS token)
        L_token_out = d_numerical + (len(categories) if categories else 0)
        
        # d_in 是 MLPDiffusion 期望的输入维度，即展平后的所有 tokens
        d_in = d_token * L_token_out 
        self.mlp = MLPDiffusion(d_in, dim_t=dim_t, use_mlp=use_mlp)
        
        self.decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)
        
        self.model = nn.ModuleList([self.tokenizer, self.encoder, self.mlp, self.decoder, self.detokenizer])

    def forward(self, x_num, x_cat, timesteps):
        # 1. 获取序列长度 L 和批量大小 B
        # 假设输入形状为 (L, B, C)
        L, B, C_num = x_num.shape
        C_cat = x_cat.shape[2] if x_cat is not None else 0

        # 2. 展平 L 和 B 维度: (L, B, C) -> (L*B, C)
        # 这使得每个时间步的数据被视为一个独立的表格样本。
        x_num_flat = x_num.reshape(L * B, C_num)
        x_cat_flat = x_cat.reshape(L * B, C_cat) if x_cat is not None else None

        # 3. Tokenizer: [L*B, L_token, d_token] (L_token = d_numerical + len(categories) + 1)
        e = self.tokenizer(x_num_flat, x_cat_flat)
        
        # 4. Encoder Input: [L*B, L_token - 1, d_token] (忽略 [CLS] token)
        decoder_input = e[:, 1:, :]  
        
        # 5. Encoder: [L*B, L_token - 1, d_token]
        y = self.encoder(decoder_input)
        
        # 6. MLPDiffusion: 展平 (L_token - 1) 维度
        # 将 timesteps (shape B) 复制 L 次以匹配新的批量大小 (L*B)
        timesteps_repeated = timesteps.repeat_interleave(B)
        
        # y.reshape(L*B, -1) -> [L*B, (L_token - 1) * d_token]
        pred_y_flat = self.mlp(y.reshape(L * B, -1), timesteps_repeated)
        
        # 7. Decoder: Reshape back to [L*B, L_token - 1, d_token]
        pred_e = self.decoder(pred_y_flat.reshape(*y.shape))
        
        # 8. Detokenizer
        # x_num_pred_flat: [L*B, d_numerical]
        # x_cat_pred_list_flat: List of [L*B, cat_size]
        x_num_pred_flat, x_cat_pred_list_flat = self.detokenizer(pred_e)
        
        # 9. 重塑输出: (L*B, C_out) -> (L, B, C_out)
        
        # 数值特征: [L, B, d_numerical]
        x_num_pred = x_num_pred_flat.reshape(L, B, -1)
        
        # 分类特征: [L, B, sum(categories)]
        if len(x_cat_pred_list_flat) > 0:
            x_cat_pred_flat = torch.cat(x_cat_pred_list_flat, dim=-1)
            x_cat_pred = x_cat_pred_flat.reshape(L, B, -1)
        else:
            # 如果没有分类特征，返回一个形状匹配的零张量
            x_cat_pred = torch.zeros((L, B, 0), dtype=x_num_pred.dtype, device=x_num_pred.device)

        # 确保 x_cat_pred 的 dtype 与 x_num_pred 匹配，正如原始代码末尾所做。
        if x_cat_pred.dtype != x_num_pred.dtype:
             x_cat_pred = x_cat_pred.to(x_num_pred.dtype)

        return x_num_pred, x_cat_pred


# ==============================================================================
# Model 和 Precond (适配 L*B 批量大小)
# ==============================================================================

class Precond(nn.Module):
    def __init__(self,
        denoise_fn,
        sigma_data = 0.5,            # Expected standard deviation of the training data.
        net_conditioning = "sigma",
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.net_conditioning = net_conditioning
        self.denoise_fn_F = denoise_fn

    def forward(self, x_num, x_cat, t, sigma):
        # 假设 x_num, x_cat 的 shape 是 (L, B, C)
        L, B = x_num.shape[0], x_num.shape[1]
        
        # 扩展 t 和 sigma 到 L*B 的批量大小
        t_repeated = t.repeat_interleave(L)
        
        # sigma: [B, 1] -> [L*B, 1]
        if sigma.dim() == 2:
            sigma_repeated = sigma.repeat_interleave(L, dim=0)
        # sigma: [B] -> [L*B]
        elif sigma.dim() == 1:
            sigma_repeated = sigma.repeat_interleave(L)
        else:
            sigma_repeated = sigma # 标量或其他情况

        x_num = x_num.to(torch.float32)

        # 在此处，sigma 可能是 [L*B] 或 [L*B, 1]
        sigma = sigma_repeated.to(torch.float32)
        
        # 同样，x_num 和 x_cat 必须展平为 (L*B, C) 才能传入 denoise_fn_F (UniModMLP)
        C_num = x_num.shape[2]
        C_cat = x_cat.shape[2] if x_cat is not None else 0
        x_num_flat = x_num.reshape(L * B, C_num)
        x_cat_flat = x_cat.reshape(L * B, C_cat) if x_cat is not None else None

        # sigma_cond 的计算基于扩展后的 sigma/t
        if sigma.dim() > 1 and sigma.shape[1] > 1: # 如果是 learnable column-wise noise schedule，则需要重新检查形状
             # 这里假设 sigma 是 [L*B, 1] 或 [L*B]
             # 如果原始 sigma 是 [B, C_features]，那么重复后是 [L*B, C_features]
             # 但原代码只检查了 sigma.dim() > 1，且后续计算是针对单个值或列的。
             # 我们假设 sigma 最终会是 [L*B] 或 [L*B, 1]
             # 为了安全，使用 t 进行 conditioning
            sigma_cond = (0.002 ** (1/7) + t_repeated * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
        else:
            sigma_cond = sigma  
        
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma_cond.log() / 4
        
        # 确保 c_in, c_skip, c_out 能够正确广播
        if c_in.dim() < x_num_flat.dim():
            c_in = c_in.unsqueeze(-1)
            c_skip = c_skip.unsqueeze(-1)
            c_out = c_out.unsqueeze(-1)

        x_in = c_in * x_num_flat
        
        # UniModMLP 期望 (L*B, C) 的输入
        if self.net_conditioning == "sigma":
            F_x_flat, x_cat_pred_flat = self.denoise_fn_F(
                x_in.reshape(L, B, C_num), 
                x_cat_flat.reshape(L, B, C_cat) if x_cat_flat is not None else None, 
                c_noise.flatten()
            )
        elif self.net_conditioning == "t":
            F_x_flat, x_cat_pred_flat = self.denoise_fn_F(
                x_in.reshape(L, B, C_num), 
                x_cat_flat.reshape(L, B, C_cat) if x_cat_flat is not None else None, 
                t_repeated
            )

        # F_x_flat 和 x_cat_pred_flat 此时是 (L, B, C) 形状，需要再次展平
        F_x_flat = F_x_flat.reshape(L * B, -1)
        x_cat_pred_flat = x_cat_pred_flat.reshape(L * B, -1)
        
        assert F_x_flat.dtype == dtype
        
        # 重新展平 x_num 到 (L*B, C)
        x_num_flat = x_num.reshape(L * B, C_num)
        
        D_x_flat = c_skip * x_num_flat + c_out * F_x_flat.to(torch.float32)
        
        # 重塑回 (L, B, C)
        D_x = D_x_flat.reshape(L, B, C_num)
        x_cat_pred = x_cat_pred_flat.reshape(L, B, -1)
        
        return D_x, x_cat_pred
    

class Model(nn.Module):
    def __init__(
            self, denoise_fn,
            sigma_data=0.5, 
            precond=False, 
            net_conditioning="sigma",
            **kwargs
        ):
        super().__init__()
        self.precond = precond
        if precond:
            self.denoise_fn_D = Precond(
                denoise_fn,
                sigma_data=sigma_data,
                net_conditioning=net_conditioning
            )
        else:
            self.denoise_fn_D = denoise_fn

    def forward(self, x_num, x_cat, t, sigma=None):
        # x_num, x_cat: (L, B, C), t: (B), sigma: (B) 或 (B, 1)
        
        if self.precond:
            # Precond 内部会处理 (L, B, C) -> (L*B, C) 的转换
            return self.denoise_fn_D(x_num, x_cat, t, sigma)
        else:
            # 如果没有 Precond，我们需要在这里处理 t 的维度适配
            L = x_num.shape[0]
            t_repeated = t.repeat_interleave(L)
            
            # UniModMLP 内部会处理展平，这里直接传入 (L, B, C) 和 (L*B) 的 t
            # NOTE: UniModMLP.forward 期望 t 是 (B) 形状，但在 UniModMLP.forward 内部被重复。
            # 为了保持 UniModMLP 调用的接口不变，这里仍传入 (B) 形状的 t。
            # 实际上，如果 Model 传入 (B) 形状的 t，而 Precond 传入 (L*B) 形状的 t_repeated，
            # 那么 UniModMLP 需要调整其内部的 t/timesteps 处理逻辑。
            
            # 保持 Model 对 UniModMLP (denoise_fn) 的调用接口不变，
            # 依赖 UniModMLP 内部的逻辑来处理序列维度。
            return self.denoise_fn_D(x_num, x_cat, t)