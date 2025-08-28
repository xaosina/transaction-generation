import torch
from scipy.optimize import linear_sum_assignment

def r1_valid(pred, true, valid_mask):
    # For inspiration: https://lightning.ai/docs/torchmetrics/stable/regression/r2_score.html
    if true.ndim == 3:
        valid_mask = valid_mask.unsqueeze(-1)
    res = (pred - true).abs()  # L, B, [D]
    userwise_res = torch.where(valid_mask, res, torch.nan).nansum(dim=0)  # B, [D]
    userwise_median = torch.where(valid_mask, true, torch.nan).nanmedian(dim=0)[0]  # B, [D]
    tot = (true - userwise_median).abs()  # L, B, [D]
    userwise_tot = torch.where(valid_mask, tot, torch.nan).nansum(dim=0)  # B, [D]
    eps = 1e-8
    rse = userwise_res / (userwise_tot + eps)
    return rse.sum(), rse.numel()

def r1(pred, true):
    # For inspiration: https://lightning.ai/docs/torchmetrics/stable/regression/r2_score.html
    res = (pred - true).abs()  # L, B, [D]
    userwise_res = res.sum(dim=0)  # B, [D]

    userwise_median = true.median(dim=0)[0]  # B, [D]
    tot = (true - userwise_median).abs()  # L, B, [D]
    userwise_tot = tot.sum(dim=0)  # B, [D]
    eps = 1e-8
    rse = userwise_res / (userwise_tot + eps)
    return rse.sum(), rse.numel()

def rse_valid(pred, true, valid_mask):
    # For inspiration: https://lightning.ai/docs/torchmetrics/stable/regression/r2_score.html
    if true.ndim == 3:
        valid_mask = valid_mask.unsqueeze(-1)
    res = (pred - true) ** 2  # L, B, [D]
    userwise_res = torch.where(valid_mask, res, torch.nan).nansum(dim=0)  # B, [D]
    userwise_mean = torch.where(valid_mask, true, torch.nan).nanmean(dim=0)  # B, [D]
    tot = (true - userwise_mean) ** 2  # L, B, [D]
    userwise_tot = torch.where(valid_mask, tot, torch.nan).nansum(dim=0)  # B, [D]
    eps = 1e-8
    rse = userwise_res / (userwise_tot + eps)
    return rse.sum(), rse.numel()


def rse(pred, true):
    # For inspiration: https://lightning.ai/docs/torchmetrics/stable/regression/r2_score.html
    res = (pred - true) ** 2  # L, B, [D]
    userwise_res = res.sum(dim=0)  # B, [D]

    userwise_mean = true.mean(dim=0)  # B, [D]
    tot = (true - userwise_mean) ** 2  # L, B, [D]
    userwise_tot = tot.sum(dim=0)  # B, [D]
    eps = 1e-8
    rse = userwise_res / (userwise_tot + eps)
    return rse.sum(), rse.numel()


def batch_linear_assignment(cost):
    b, w, t = cost.shape
    matching = torch.full([b, w], -1, dtype=torch.long, device=cost.device)
    for i in range(b):
        workers, tasks = linear_sum_assignment(
            cost[i].numpy(), maximize=False
        )  # (N, 2).
        workers = torch.from_numpy(workers)
        tasks = torch.from_numpy(tasks)
        matching[i].scatter_(0, workers, tasks)
    return matching
