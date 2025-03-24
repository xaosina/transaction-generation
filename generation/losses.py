from typing import Optional
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from generation.data.data_types import Batch, PredBatch

@dataclass
class LossConfig:
    model: Optional[str] = "baseline"
    c_dim: Optional[int] = None
    c_number: Optional[int] = None

def get_loss(config: LossConfig):

    model = config.model

    if model == "baseline":
        return BaselineLoss()
    else:
        raise ValueError(f"Unknown type of target (target_type): {model}")


class BaselineLoss:
    def __init__(self,):
        super().__init__()

    def _compute_loss(self, y_true: Batch, y_pred: PredBatch):
        mse = .0
        mse_total = 0
        if y_pred.time is not None:
            mse += F.mse_loss(y_pred.time, y_true.time)
            mse_total += 1
        breakpoint()

        if y_pred.num_features is not None:
            mse += F.mse_loss(y_pred.num_features, y_true.num_features[:, :, 1:]) # [:, :, 1:] - time doesn't count
            mse_total += 1
        
        mse = mse / mse_total
        
        ce = .0
        ce_total = 0
        if y_pred.cat_features is not None:
            for key in y_pred.cat_features:
                breakpoint()
                ce += F.cross_entropy(
                    y_pred.cat_features[key].permute(1, 2, 0), 
                    y_true[key].permute(1, 0)
                    )
                ce_total += 1
            assert ce_total != .0
            ce = ce / ce_total
        
        return mse + ce
    
    def __call__(self, *args, **kwds):
        return self._compute_loss(*args, **kwds)
