import torch 
import torch.nn.functional as F 
import numpy as np


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x


def log_onehot_to_index(log_e):
    return log_e.argmax(1)


def log_sample_categorical(logits,num_classes):
    ## use Gumbel-Max Trick to logits, then get index and convert to log one-hot
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample = (gumbel_noise + logits).argmax(dim=1)
    log_sample = index_to_log_onehot(sample,num_classes)
    return log_sample

def extend_multi_classes(a, b):
    
    data_cat = []

    for i in b:
        c = a - np.log(i)
        c = c.repeat(1,i,1)
        data_cat.append(c)

    final_tensor = torch.cat(data_cat, dim=1) # dim=1 is likely correct for horizontal extension

    return final_tensor

def log_sample_categorical_multi_task(logits, num_classes_list):
    #
    """
    使用 Gumbel-Max Trick 从多任务 Logits 中采样，并返回 Log-One-Hot 格式的样本。

    Args:
        logits (torch.Tensor): 模型的 Logits,形状为 (B, Sum_C, L)。
        num_classes_list (list[int]): 每个独立分类任务的类别数 [C1, C2, ..., CD]。

    Returns:
        torch.Tensor: Log-One-Hot 格式的样本，形状为 (B, Sum_C, L)。
    """
    sum_c = sum(num_classes_list)
    assert logits.size(1) == sum_c, "Logits dim 1 must match sum of classes."
    
    B, Sum_C, L = logits.shape
    D = len(num_classes_list) # 任务数量
    
    log_sample_parts = []
    start_idx = 0

    # 1. 遍历 D 个任务
    for i in range(D):
        C_i = num_classes_list[i]
        
        # 2. 切片：获取当前任务的 Logits，形状: (B, C_i, L)
        logits_task = logits[:, start_idx : start_idx + C_i, :]
        
        # 3. Gumbel-Max Trick 采样 (Gumbel-Max 必须在类别维度上进行)
        
        # a. 生成 Gumbel 噪声，形状 (B, C_i, L)
        uniform = torch.rand_like(logits_task)
        # Numerical stability: -log(-log(uniform))
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        
        # b. Gumbel Noise + Logits
        gumbel_logits = gumbel_noise + logits_task
        
        # c. Argmax：在类别维度 (dim=1) 上找到最大值，得到索引
        # 结果形状: (B, L)
        sample_index = gumbel_logits.argmax(dim=1)
        
        # 4. 转换为 Log-One-Hot
        # 使用辅助函数，将 (B, L) 索引转换为 (B, C_i, L) Log-One-Hot 
        log_sample_task = index_to_log_onehot(sample_index, C_i)
        
        log_sample_parts.append(log_sample_task)
        start_idx += C_i
        
    # 5. 拼接：沿着类别/特征维度 (dim=1) 拼接，最终形状: (B, Sum_C, L)
    return torch.cat(log_sample_parts, dim=1)


def log_onehot_to_index_multi_task(log_e, num_classes_list):
    #
    """
    将多任务 Log-One-Hot/Logits 张量 (B, Sum_C, L) 转换回原始的索引张量 (B, L, D)。

    Args:
        log_e (torch.Tensor): Logits 或 Log-One-Hot 张量，形状为 (B, Sum_C, L)。
        num_classes_list (list[int]): 每个独立分类任务的类别数列表 [C1, C2, ..., CD]。

    Returns:
        torch.Tensor: 原始的类别索引张量，形状为 (B, L, D)。
    """
    # 验证输入维度
    sum_c = sum(num_classes_list)
    assert log_e.size(1) == sum_c, \
        f"Input dim 1 ({log_e.size(1)}) must match sum of classes ({sum_c})"
    
    B, Sum_C, L = log_e.shape
    D = len(num_classes_list) # 任务数量
    
    index_parts = []
    start_idx = 0

    # 1. 遍历 D 个任务
    for i in range(D):
        C_i = num_classes_list[i]
        
        # 2. 切片：获取当前任务的 Logits/Log-One-Hot
        # 形状: (B, C_i, L)
        log_e_task = log_e[:, start_idx : start_idx + C_i, :]
        
        # 3. 独立 Argmax：沿着类别维度 (dim=1) 找到最高概率的索引
        # 结果形状: (B, 1, L)
        indices_task = log_e_task.argmax(dim=1, keepdim=True)
        
        # 4. 调整维度：从 (B, 1, L) 转换为 (B, L, 1)
        # 这就是 (B, L, D) 形状中的一个任务索引部分
        # permute(0, 2, 1) -> (B, L, 1)
        indices_task = indices_task.permute(0, 2, 1)
        
        index_parts.append(indices_task)
        start_idx += C_i
        
    # 5. 拼接：沿着最后一个维度 (dim=2) 拼接所有任务的索引
    # 最终形状: (B, L, D)
    return torch.cat(index_parts, dim=2)


def index_to_log_onehot_multi_task(x, num_classes_list):
    #
    """
    Converts an index tensor (B, L, D) into a log-one-hot tensor for multi-task classification.

    Args:
        x (torch.Tensor): The input tensor of integer indices, shape (B, L, D).
        num_classes_list (list[int]): A list of class counts, where len(list) == D.

    Returns:
        torch.Tensor: The log-one-hot tensor, shape (B, Total_Classes, L),
                      where Total_Classes = sum(num_classes_list).
    """
    B, L, D = x.shape
    
    # 1. Assertion Check
    assert D == len(num_classes_list), \
        f"Input dim D ({D}) must match length of num_classes_list ({len(num_classes_list)})"

    log_onehot_parts = []
    
    # Iterate over the D tasks
    for i in range(D):
        task_num_classes = num_classes_list[i]
        
        # Select the slice for the current task: shape (B, L)
        x_task = x[..., i] 

        # 2. Index Clipping/Adjustment (using .clone() for safety)
        # Check for indices that are exactly equal to the max class index
        x_task_adjusted = x_task.clone()
        #x_task_adjusted[x_task_adjusted == task_num_classes] = task_num_classes - 1

        # Final check to ensure no index is out of bounds
        assert x_task_adjusted.max().item() < task_num_classes, \
            f'Task {i} Error: {x_task_adjusted.max().item()} >= {task_num_classes}'

        # 3. One-Hot Encoding: shape (B, L, task_num_classes)
        x_onehot = F.one_hot(x_task_adjusted, task_num_classes)
        x_onehot = x_onehot.permute(0,2,1)
        # 4. Log and Clamp
        log_x = torch.log(x_onehot.float().clamp(min=1e-30))
        
        log_onehot_parts.append(log_x)
    # 5. Concatenate all log-one-hot parts along the class dimension (dim=1)
    # Resulting shape: (B, Total_Classes, L)
    return torch.cat(log_onehot_parts, dim=1)

def softmax_to_logits_multi_task(y_logits, num_classes_list):
    #
    """
    将模型输出 (B,Sum_Classes,L) 拆分，并在每个任务的类别维度上独立应用 log_softmax。
    
    Args:
        y_logits (torch.Tensor): 模型输出的 logits, 形状 (B,Sum_Classes, L)。
        num_classes_list (list[int]): 每个任务的类别数 [C1, C2, ..., CD]。
        
    Returns:
        torch.Tensor: 形状为 (B,Sum_Classes, L) 的对数概率张量。
    """
    
    start_idx = 0
    log_probs_parts = []

    for C_i in num_classes_list:
        # 1. 切片：获取当前任务的 logits (B,C_i,L)
        y_task_logits = y_logits[:, start_idx : start_idx + C_i,:]
        
        # 2. 独立应用 log_softmax
        # dim=1 对应于当前切片的类别维度 C_i
        log_probs_task = F.log_softmax(y_task_logits, dim=1)
        
        log_probs_parts.append(log_probs_task)
        start_idx += C_i
        
    # 3. 沿特征维度拼接回 (B,Sum_Classes,l)
    return torch.cat(log_probs_parts, dim=1)