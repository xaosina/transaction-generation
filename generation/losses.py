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
    def __init__(
        self,
    ):
        super().__init__()
        self.ignore_index = -100

    def _compute_loss(self, y_true: Batch, y_pred: PredBatch) -> torch.Tensor:
        padding_mask = (
            torch.arange(y_true.time.shape[0], device=y_true.time.device)
            >= y_pred.lengths[:, None]
        )
        mse = 0.0
        mse_total = 0
        if y_pred.time is not None:
            pred_time = y_pred.time.permute(1, 0) * padding_mask
            true_time = y_true.time.permute(1, 0) * padding_mask
            mse += F.mse_loss(
                pred_time[:, :-1],
                true_time[:, 1:],
            )
            mse_total += 1

        if y_pred.num_features is not None:
            pred_num = y_pred.num_features.permute(1, 0, 2) * padding_mask.unsqueeze(-1)
            true_num = y_true.num_features.permute(1, 0, 2) * padding_mask.unsqueeze(-1)
            mse += F.mse_loss(
                pred_num[:, :-1, :],
                true_num[:, 1:, 1:],  # second slice - time doesn't count
            )
            mse_total += 1

        mse = mse / mse_total

        ce = 0.0
        ce_total = 0
        breakpoint()
        if y_pred.cat_features is not None:
            for key in y_pred.cat_features:
                true_cat = y_true[key].permute(1, 0)
                true_cat[~padding_mask] = self.ignore_index
                
                true_num = y_pred.cat_features[key].permute(1, 2, 0)
                
                ce += F.cross_entropy(
                    true_num[:, :, :-1],
                    true_cat[:, 1:],
                    ignore_index=self.ignore_index,
                )
                ce_total += 1
            assert ce_total != 0.0
            ce = ce / ce_total

        return mse + ce

    def __call__(self, *args, **kwds):
        return self._compute_loss(*args, **kwds)
