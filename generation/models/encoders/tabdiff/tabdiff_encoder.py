import torch

import torch.nn.functional as F
from .noise_scheduler import *
#from .diffusion_utils import UniModMLP,Model
from .unet_utils import Model,Unet1DDiffusion
from torch import nn
from itertools import chain
import numpy as np

from .transformer_decoder import Transformer_denoise

S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1


class ConditionalContinuousDiscreteEncoder(nn.Module):

    def __init__(self,data_conf,model_config):
        super().__init__()
        self.d_weight,self.c_weight = 1.0,1.0
        device = "cuda:0"
        model_params = model_config.params
        num_numerical_features = len(data_conf.num_names)+1
        #num_classes = list(data_conf.cat_cardinalities.values())
        ## add 1,because of mask index
        #num_classes = torch.tensor([5,204])
        num_classes = torch.tensor(list(data_conf.cat_cardinalities.values()))
        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device) if len(num_classes)>0 else torch.tensor([]).to(device).int()
        self.mask_index = torch.tensor(self.num_classes).long().to(device)
        self.neg_infinity = -1000000.0 
        self.num_classes_w_mask = tuple(self.num_classes + 1)

        offsets = np.cumsum(self.num_classes)
        offsets = np.append([0], offsets)
        self.slices_for_classes = []
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(offsets).to(device)
        
        offsets = np.cumsum(self.num_classes) + np.arange(1, len(self.num_classes)+1)
        offsets = np.append([0], offsets)
        self.slices_for_classes_with_mask = []
        for i in range(1, len(offsets)):
            self.slices_for_classes_with_mask.append(np.arange(offsets[i - 1], offsets[i]))

        # backbone = Transformer_denoise(d_numerical=num_numerical_features,
        #                            categories=self.num_classes_w_mask,
        #                            **model_params['unet_params'])

        backbone = Unet1DDiffusion(d_numerical=num_numerical_features,
                                   categories=self.num_classes_w_mask,
                                   **model_params['unet_params'])
        
        model = Model(backbone, **model_params['edm_params'])
        self._denoise_fn = model
        
        self.num_timesteps = 500
        self.closs_weight_schedule = "anneal"
        self.scheduler = model_params['scheduler']
        self.cat_scheduler = model_params['cat_scheduler']
        self.noise_dist = model_params['noise_dist']
        self.edm_params = model_params['edm_params']
        self.noise_dist_params = model_params['noise_dist_params']
        self.sampler_params = model_params['sampler_params']
        noise_schedule_params = model_params['noise_schedule_params']

        if self.num_numerical_features == 0:
            self.sampler_params['stochastic_sampler'] = True
            self.sampler_params['second_order_correction'] = True
        
        self.w_num = 0.0
        self.w_cat = 0.0
        self.num_mask_idx = []
        self.cat_mask_idx = []
        
        self.device = device
        
        if self.scheduler == 'power_mean':
            self.num_schedule = PowerMeanNoise(**noise_schedule_params)
        elif self.scheduler == 'power_mean_per_column':
            self.num_schedule = PowerMeanNoise_PerColumn(num_numerical = num_numerical_features, **noise_schedule_params)
        else:
            raise NotImplementedError(f"The noise schedule--{self.scheduler}-- is not implemented for contiuous data at CTIME ")
        
        if self.cat_scheduler == 'log_linear':
            self.cat_schedule = LogLinearNoise(**noise_schedule_params)
        elif self.cat_scheduler == 'log_linear_per_column':
            self.cat_schedule = LogLinearNoise_PerColumn(num_categories = len(num_classes), **noise_schedule_params)
        else:
            raise NotImplementedError(f"The noise schedule--{self.cat_scheduler}-- is not implemented for discrete data at CTIME ")

        self.iter = 1

    def forward(self,hist,tgt):

        
        device = "cuda:0"
        #
        ## All input tensor has shape: (L,B,F_i), we have to permute them into (B,L,F_i)

        tgt["num"],tgt["cat"] = tgt["num"].permute(1,0,2),tgt["cat"].permute(1,0,2)
        hist["num"],hist["cat"] = hist["num"].permute(1,0,2),hist["cat"].permute(1,0,2)

        bs,ls = tgt["num"].size(0),tgt["num"].size(1)
        d_num,c_num = tgt["num"].size(2),tgt["cat"].size(2)
        ## flat tgt to 2d to (B*L,F_i)
        x_num,x_cat = tgt["num"].reshape(-1,d_num),tgt["cat"].reshape(-1,c_num)

        hist["cat"] = tensor_to_one_hot(hist["cat"],self.num_classes+1)
        
        # Sample noise level for each batch, and the repeat L times to match the length of data:bs*ls
        #if self.iter >= 800:
        #    
        t = torch.rand(bs, device=device, dtype=x_num.dtype)
        t = torch.repeat_interleave(t,ls)
        t = t[:, None]
        sigma_num = self.num_schedule.total_noise(t)
        sigma_cat = self.cat_schedule.total_noise(t)
        dsigma_cat = self.cat_schedule.rate_noise(t)

        # Convert sigma_cat to the corresponding alpha and move_chance
        alpha = torch.exp(-sigma_cat)
        move_chance = -torch.expm1(-sigma_cat)      # torch.expm1 gives better numertical stability
        
        # Continuous forward diff
        x_num_t = x_num
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            #x_num_t = x_num + noise * sigma_num.unsqueeze(2)
            x_num_t = x_num + noise * sigma_num
        
        # Discrete forward diff
        x_cat_t = x_cat
        x_cat_t_soft = x_cat # in the case where x_cat is empty, x_cat_t_soft will have the same shape as x_cat

        x_cat_t, x_cat_t_soft = self.q_xt(x_cat, move_chance,strategy="soft")

        # Predict orignal data (distribution)
        
        model_out_num, model_out_cat = self._denoise_fn(   
            x_num_t, 
            x_cat_t_soft,
            t.squeeze(),
            hist,
            sigma=sigma_num,
        )

        d_loss = torch.zeros((1,)).float()
        c_loss = torch.zeros((1,)).float()

        if x_num.shape[1] > 0:
            c_loss = self._edm_loss(model_out_num, x_num, sigma_num)
        if x_cat.shape[1] > 0:
            logits = self._subs_parameterization(model_out_cat, x_cat_t)    # log normalized probabilities, with the entry mask category being set to -inf
            d_loss = self._absorbed_closs(logits, x_cat, sigma_cat, dsigma_cat)
        
        if self.closs_weight_schedule == "fixed":
            pass
        elif self.closs_weight_schedule == "anneal":
            frac_done = self.iter / 100000000
            closs_weight = self.c_weight * max((1 - frac_done),0.00001)
            self.c_weight = closs_weight
            self.iter += 1

        total_loss = d_loss.mean()*self.d_weight + c_loss.mean()*self.c_weight
        return  total_loss
        
    def q_xt(self, x, move_chance, strategy='hard'):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input. 
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        if strategy == 'hard':
            move_indices = torch.rand(
            * x.shape, device=x.device) < move_chance
            xt = torch.where(move_indices, self.mask_index, x)
            xt_soft = self.to_one_hot(xt).to(move_chance.dtype)
            return xt, xt_soft
        elif strategy == 'soft':
            bs = x.shape[0]
            xt_soft = torch.zeros(bs, torch.sum(self.mask_index+1), device=x.device)
            xt = torch.zeros_like(x)
            for i in range(len(self.num_classes)):
                slice_i = self.slices_for_classes_with_mask[i]
                # set the bernoulli probabilities, which determines the "coin flip" transition to the mask class
                prob_i = torch.zeros(bs, 2, device=x.device)
                prob_i[:,0] = 1-move_chance[:,i]
                prob_i[:,-1] = move_chance[:,i]
                log_prob_i = torch.log(prob_i)
                # draw soft samples and place them back to the corresponding columns
                soft_sample_i = F.gumbel_softmax(log_prob_i, tau=0.01, hard=True)
                idx = torch.stack((x[:,i]+slice_i[0], torch.ones_like(x[:,i])*slice_i[-1]), dim=-1)
                xt_soft[torch.arange(len(idx)).unsqueeze(1), idx] = soft_sample_i
                # retrieve the hard samples
                xt[:, i] = torch.where(soft_sample_i[:,1] > soft_sample_i[:,0], self.mask_index[i], x[:,i])
            return xt, xt_soft  
        
    def to_one_hot(self, x_cat):
        x_cat_oh = torch.cat(
            [F.one_hot(x_cat[:, i], num_classes=self.num_classes[i]+1,) for i in range(len(self.num_classes))], 
            dim=-1
        )
        return x_cat_oh

    def _edm_loss(self, D_yn, y, sigma):
        weight = (sigma ** 2 + self.edm_params['sigma_data'] ** 2) / (sigma * self.edm_params['sigma_data']) ** 2
    
        target = y
        loss = weight * ((D_yn - target) ** 2)

        return loss

    def _absorbed_closs(self, model_output, x0, sigma, dsigma):
        """
            alpha: (bs,)
        """
        log_p_theta = torch.gather(
            model_output, -1, x0[:, :, None]
        ).squeeze(-1)
        alpha = torch.exp(-sigma)
        if self.cat_scheduler in ['log_linear_unified', 'log_linear_per_column']:
            elbo_weight = - dsigma / torch.expm1(sigma)
        else:
            elbo_weight = -1/(1-alpha)
        
        loss = elbo_weight * log_p_theta
        return loss
    
    def _subs_parameterization(self, unormalized_prob, xt):
        # log prob at the mask index = - infinity
        unormalized_prob = self.pad(unormalized_prob, self.neg_infinity)
        
        unormalized_prob[:, range(unormalized_prob.shape[1]), self.mask_index] += self.neg_infinity
        
        # Take log softmax on the unnormalized probabilities to the logits
        logits = unormalized_prob - torch.logsumexp(unormalized_prob, dim=-1,
                                        keepdim=True)
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)    # (bs, K)
        logits[unmasked_indices] = self.neg_infinity 
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def pad(self, x, pad_value):
        splited = torch.split(x, self.num_classes_w_mask, dim=-1)
        max_K = max(self.num_classes_w_mask)
        padded_ = [
            torch.cat((
                t, 
                pad_value*torch.ones(*(t.shape[:-1]), max_K-t.shape[-1], dtype=t.dtype, device=t.device)
            ), dim=-1) 
        for t in splited]
        out = torch.stack(padded_, dim=-2)
        return out

    def sample(self,hist,gen_len):
        

        b,ls = hist["cat"].shape[1],hist["cat"].shape[0]
        
        hist["num"],hist["cat"] = hist["num"].permute(1,0,2),hist["cat"].permute(1,0,2)
        hist["cat"] = tensor_to_one_hot(hist["cat"],self.num_classes+1)
        
        device = self.device
        dtype = torch.float32
        
        # Create the chain of t
        t = torch.linspace(0,1,self.num_timesteps, dtype=dtype, device=device)      # times = 0.0,...,1.0
        t = t[:, None]
        
        # Compute the chains of sigma
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[0:-1]
        sigma_cat_next = torch.zeros_like(sigma_cat_cur)
        sigma_cat_next[1:] = sigma_cat_cur[0:-1]
        
        # Prepare sigma_hat for stochastic sampling mode
        if self.sampler_params['stochastic_sampler']:
            gamma = min(S_churn / self.num_timesteps, np.sqrt(2) - 1) * (S_min <= sigma_num_cur) * (sigma_num_cur <= S_max)
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.num_schedule.inverse_to_t(sigma_num_hat)
            t_hat = torch.min(t_hat, dim=-1, keepdim=True).values    # take the samllest t_hat induced by sigma_num
            zero_gamma = (gamma==0).any()
            t_hat[zero_gamma] = t[zero_gamma]
            out_of_bound = (t_hat > 1).squeeze()
            sigma_num_hat[out_of_bound] = sigma_num_cur[out_of_bound]
            t_hat[out_of_bound] = t[out_of_bound]
            sigma_cat_hat = self.cat_schedule.total_noise(t_hat)
        else:
            t_hat = t
            sigma_num_hat = sigma_num_cur
            sigma_cat_hat = sigma_cat_cur
                
        # Sample priors for the continuous dimensions
        z_norm = torch.randn((b*gen_len,self.num_numerical_features), device=device) * sigma_num_cur[-1] 
            
        # Sample priors for the discrete dimensions
        has_cat = len(self.num_classes) > 0
        z_cat = torch.zeros((b*gen_len, 0), device=device).float()      # the default values for categorical sample if the dataset has no categorical entry
        if has_cat:
            z_cat = self._sample_masked_prior(
                b*gen_len,
                len(self.num_classes),
            )

        for i in reversed(range(0, self.num_timesteps)):                  
            z_norm, z_cat, q_xs = self.edm_update(
                z_norm, z_cat, i, 
                t[i], t[i-1] if i > 0 else None, t_hat[i],
                hist,
                sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i], 
                sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
            )
        #breakpoint()
        assert torch.all(z_cat < self.mask_index)
        #sample = torch.cat([z_norm, z_cat], dim=1)
        return z_norm,z_cat
    
    def _sample_masked_prior(self, *batch_dims):
        return self.mask_index[None,:] * torch.ones(    
        * batch_dims, dtype=torch.int64, device=self.mask_index.device)
    
    def edm_update(
            self, x_num_cur, x_cat_cur, i, 
            t_cur, t_next, t_hat,
            hist,
            sigma_num_cur, sigma_num_next, sigma_num_hat, 
            sigma_cat_cur, sigma_cat_next, sigma_cat_hat, 
        ):
        """
        i = T-1,...,0
        """
        #cfg = self.y_only_model is not None
        b = x_num_cur.shape[0]
        has_cat = len(self.num_classes) > 0
        
        # Get x_num_hat by move towards the noise by a small step
        x_num_hat = x_num_cur + (sigma_num_hat ** 2 - sigma_num_cur ** 2).sqrt() * S_noise * torch.randn_like(x_num_cur)
        # Get x_cat_hat
        move_chance = -torch.expm1(sigma_cat_cur - sigma_cat_hat)    # the incremental move change is 1 - alpha_t/alpha_s = 1 - exp(sigma_s - sigma_t)
        x_cat_hat, _ = self.q_xt(x_cat_cur, move_chance) if has_cat else (x_cat_cur, x_cat_cur)

        # Get predictions
        x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_hat.dtype) if has_cat else x_cat_hat
        denoised, raw_logits = self._denoise_fn(
            x_num_hat.float(), x_cat_hat_oh,
            t_hat.squeeze().repeat(b),hist, sigma=sigma_num_hat.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
        )
        
        # Euler step
        d_cur = (x_num_hat - denoised) / sigma_num_hat
        x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * d_cur
        
        # Unmasking
        x_cat_next = x_cat_cur
        q_xs = torch.zeros_like(x_cat_cur).float()
        if has_cat:
            logits = self._subs_parameterization(raw_logits, x_cat_hat)
            alpha_t = torch.exp(-sigma_cat_hat).unsqueeze(0).repeat(b,1)
            alpha_s = torch.exp(-sigma_cat_next).unsqueeze(0).repeat(b,1)
            x_cat_next, q_xs = self._mdlm_update(logits, x_cat_hat, alpha_t, alpha_s)
        
        # Apply 2nd order correction.
        if self.sampler_params['second_order_correction']:
            if i > 0:
                x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_next.dtype) if has_cat else x_cat_hat
                denoised, raw_logits = self._denoise_fn(
                    x_num_next.float(), x_cat_hat_oh,
                    t_next.squeeze().repeat(b),hist, sigma=sigma_num_next.unsqueeze(0).repeat(b,1)
                )
                
                d_prime = (x_num_next - denoised) / sigma_num_next
                x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        return x_num_next, x_cat_next, q_xs
    
    def _mdlm_update(self, log_p_x0, x, alpha_t, alpha_s):
        """
            # t: (bs,)
            log_p_x0: (bs, K, K_max)
            # alpha_t: (bs,)
            # alpha_s: (bs,)
            alpha_t: (bs, 1/K_cat)
            alpha_s: (bs,1/K_cat)
        """
        move_chance_t = 1 - alpha_t
        move_chance_s = 1 - alpha_s     
        move_chance_t = move_chance_t.unsqueeze(-1)
        move_chance_s = move_chance_s.unsqueeze(-1)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        # There is a noremalizing term is (1-\alpha_t) who's responsility is to ensure q_xs is normalized. 
        # However, omiting it won't make a difference for the Gumbel-max sampling trick in  _sample_categorical()
        q_xs = log_p_x0.exp() * (move_chance_t
                                - move_chance_s)
        q_xs[:, range(q_xs.shape[1]), self.mask_index] = move_chance_s[:, :, 0]
        
        # Important: make sure that prob of dummy classes are exactly 0
        dummy_mask = torch.tensor([[(1 if i <= mask_idx else 0) for i in range(max(self.mask_index+1))] for mask_idx in self.mask_index], device=q_xs.device)
        dummy_mask = torch.ones_like(q_xs) * dummy_mask
        q_xs *= dummy_mask
        
        _x = self._sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        
        z_cat = copy_flag * x + (1 - copy_flag) * _x
        return copy_flag * x + (1 - copy_flag) * _x, q_xs
    
    def _sample_categorical(self, categorical_probs):
        gumbel_norm = (
            1e-10
            - (torch.rand_like(categorical_probs) + 1e-10).log())
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    
def tensor_to_one_hot(X_index, categories) -> torch.Tensor:
    """
    将包含类别索引的张量 (L, B, D) 转换为 One-Hot 编码张量。
    D 维中的每个特征可以有不同的类别数量。

    Args:
        X_index: 形状为 (L, B, D) 的 LongTensor，其中 D 是类别特征的数量，
                 包含每个样本的类别索引。
        categories: 一个列表，长度等于 D (X_index.shape[-1])，
                    其中 categories[i] 是第 i 个特征的类别总数。

    Returns:
        One-Hot 编码后的张量，形状为 (L, B, sum(categories))。
    """
    
    # 检查输入张量是否为整数类型 (One-Hot 编码要求)
    if X_index.dtype not in [torch.long, torch.int]:
        raise TypeError(f"输入张量的数据类型必须是整数 (torch.long 或 torch.int)，但当前是 {X_index.dtype}")

    # 检查 D 维度是否与 categories 列表长度匹配
    D = X_index.shape[-1]
    if D != len(categories):
        raise ValueError(
            f"张量的最后一个维度 D ({D}) 必须与 categories 列表的长度 ({len(categories)}) 匹配。"
        )

    # 确保张量在内存中是连续的，以进行切片操作（可选，但推荐）
    X_index = X_index.contiguous()

    one_hot_list = []

    # 逐特征 (D 维度) 进行 One-Hot 编码
    for i in range(D):
        # 1. 提取第 i 个特征的索引。形状为 (L, B)
        feature_indices = X_index[..., i] 
        
        # 2. 获取当前特征的类别数
        num_classes = categories[i]
        
        # 3. 进行 One-Hot 编码
        # F.one_hot 的结果是 (L, B, num_classes)
        # 转换为 float，以便后续的嵌入/计算
        one_hot_feature = F.one_hot(feature_indices, num_classes=num_classes).float() 
        
        one_hot_list.append(one_hot_feature)

    # 4. 拼接 (Concatenation)
    # 沿最后一个维度（类别维度）拼接所有 One-Hot 结果
    X_one_hot_final = torch.cat(one_hot_list, dim=-1)

    return X_one_hot_final

def one_hot_to_tensor(X_one_hot: torch.Tensor, categories) -> torch.Tensor:
    """
    将 One-Hot 编码张量 (L, B, Sum(Categories)) 转换回类别索引张量 (L, B, D)。
    D 维中的每个特征可以有不同的类别数量。

    Args:
        X_one_hot: 形状为 (L, B, Sum(Categories)) 的浮点张量，是 One-Hot 编码后的数据。
        categories: 一个列表，长度等于 D (原始特征的数量)，
                    其中 categories[i] 是第 i 个特征的类别总数。
                    这个列表用于确定在 One-Hot 张量中如何切分出各个特征的块。

    Returns:
        包含类别索引的 LongTensor，形状为 (L, B, D)。
    """
    
    # --- 1. 输入校验 ---
    # 通常 One-Hot 编码使用浮点类型，但我们只给出警告，不强制中断
    if X_one_hot.dtype not in [torch.float32, torch.float64]:
        print(f"警告：输入 One-Hot 张量的数据类型是 {X_one_hot.dtype}。建议使用浮点类型。")

    D_one_hot = X_one_hot.shape[-1]
    expected_D_one_hot = sum(categories)
    
    if D_one_hot != expected_D_one_hot:
        raise ValueError(
            f"One-Hot 张量的最后一个维度 ({D_one_hot}) 与 'categories' 列表定义的总类别数 ({expected_D_one_hot}) 不匹配。"
        )

    if not categories:
        raise ValueError("categories 列表不能为空。")

    # --- 2. 解码过程 ---
    index_list = []
    current_offset = 0

    # 逐特征 (D 维度) 进行 One-Hot 反向解码
    for i, num_classes in enumerate(categories):
        # 1. 确定当前特征的 One-Hot 块的起始和结束位置
        start_index = current_offset
        end_index = current_offset + num_classes
        
        # 2. 提取第 i 个特征的 One-Hot 块。形状为 (L, B, num_classes)
        # One-Hot 块是从总 One-Hot 维度上进行切片
        one_hot_feature_block = X_one_hot[..., start_index:end_index] 
        
        # 3. 找到 One-Hot 编码中值为 1 的索引，即原始的类别索引。形状为 (L, B)
        # torch.argmax 默认返回 LongTensor (长整数)，适合作为索引
        feature_indices = torch.argmax(one_hot_feature_block, dim=-1)
        
        # 4. 将索引张量添加到列表中
        index_list.append(feature_indices)
        
        # 5. 更新偏移量，指向下一个特征的起始位置
        current_offset = end_index

    # --- 3. 拼接 (Stack) ---
    # 沿新的维度 (D 维度) 拼接所有索引结果。
    # 结果形状从 (L, B) * D 堆叠为 (L, B, D)
    X_index_final = torch.stack(index_list, dim=-1)

    return X_index_final