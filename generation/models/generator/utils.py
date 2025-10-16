import torch
from ...data.data_types import GenBatch, LatentDataConfig

try:
    from torch_linear_assignment import batch_linear_assignment
except ImportError:
    print("Using slow linear assignment implementation")
    from .utils import batch_linear_assignment


def match_seqs(reference: GenBatch, target: GenBatch, data_conf: LatentDataConfig):
    assert reference.shape == target.shape
    assert (reference.lengths == target.lengths).all()
    L, B = reference.shape

    cost = torch.zeros((B, L, L), device=reference.device)

    # 1. Compute cat
    for name in data_conf.focus_cat:
        pred = reference[name].T  # B, L
        true = target[name].T  # B, L
        cost += pred[:, :, None] != true[:, None, :]  # B, L, L

    # 2. Compute num
    for name in data_conf.focus_num:
        if name == data_conf.time_name:
            pred = reference.time.T
            true = target.time.T
        else:
            pred = reference[name].T
            true = target[name].T
        denominator = torch.abs(true - torch.median(true, 1)[0][:, None])  # B, L
        denominator = denominator.sum(axis=1).mean()
        nominator = torch.abs(pred[:, :, None] - true[:, None, :])  # [B, L, L]
        # if nominator.sum() == 0:
        #     denominator[nominator.sum(0).sum(0) == 0] = 1
        # nominator[:, :, (denominator == 0)] = 1 / gen_len
        # denominator[denominator == 0] = 1

        cost += nominator / denominator

    if cost.isnan().any():
        return torch.tensor(float("nan"))
    assignment = batch_linear_assignment(cost).T  # L, B
    for attr in ["time", "num_features", "cat_features"]:
        field = getattr(target, attr)
        if isinstance(field, torch.Tensor):
            assign = assignment.unsqueeze(-1) if field.ndim > 2 else assignment
            field = field.take_along_dim(assign, 0)
        setattr(target, attr, field)
    return target

def post_process_generation(pred: GenBatch, hist: GenBatch, linear_clip=False):
    if not hist.monotonic_time:
        pred.time = pred.time.clip(min=0)
    else:
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
        # Clipping
        last_time = hist.tail(1).time  # 1, B
        if not linear_clip:  # Simple Clip
            mask = pred.time < last_time
            pred.time[mask] = last_time.expand(pred.time.shape)[mask]
        else:
            L, B = pred.time.shape
            time = pred.time
            lt = last_time.expand(L, B)
            k = (time <= lt).sum(dim=0)  # Number of bad times
            i = torch.arange(L, device=time.device).view(L, 1)
            mask = i < k
            t_end = time.gather(0, k.clamp(max=L - 1).view(1, B))  # First good time
            t_end = torch.where(k == L, last_time, t_end)  # If all times bad
            pred.time = torch.where(
                mask,
                last_time
                + ((i + 1).float() / (k.clamp(min=1) + 1)) * (t_end - last_time),
                time,
            )  # Linear approximation from last time to first greater time.
    return pred