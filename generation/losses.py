from typing import Optional
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from generation.data.data_types import Batch, PredBatch


@dataclass
class LossConfig:
    name: Optional[str] = "baseline"
    c_dim: Optional[int] = None
    c_number: Optional[int] = None


def get_loss(config: LossConfig):

    name = config.name

    if name == "baseline":
        return BaselineLoss()
    else:
        raise ValueError(f"Unknown type of target (target_type): {name}")


class BaselineLoss:
    def __init__(
        self,
    ):
        super().__init__()
        self.ignore_index = -100

    def __call__(self, y_true: Batch, y_pred: PredBatch) -> torch.Tensor:
        valid_mask = torch.arange(y_true.lengths.max())[:, None] < (
            y_true.lengths
        )  # [L, B]

        mse = 0.0
        mse_count = 0

        if y_pred.time is not None:
            pred_time = y_pred.time
            true_time = y_true.time
            mse_time = F.mse_loss(pred_time[:-1], true_time[1:], reduction="none")[
                valid_mask[1:]
            ]

            mse += mse_time.sum()
            mse_count += valid_mask[1:].sum()

        if y_pred.num_features is not None:
            num_feature_ids = [
                y_true.num_features_names.index(name)
                for name in y_pred.num_features_names
            ]

            pred_num = y_pred.num_features
            true_num = y_true.num_features

            mse_num = F.mse_loss(
                pred_num[:-1],
                true_num[1:, :, num_feature_ids],
                reduction="none"
            ) * valid_mask[1:].unsqueeze(-1)

            mse += mse_num.sum()
            mse_count += mse_num.numel()

        mse = mse / mse_count

        ce = 0.0
        ce_total = 0
        if y_pred.cat_features is not None:
            for key in y_pred.cat_features:
                true_cat = y_true[key]
                true_cat[~valid_mask] = self.ignore_index
                true_cat = true_cat.permute(1, 0)

                true_num = y_pred.cat_features[key].permute(1, 2, 0)

                ce += F.cross_entropy(
                    true_num[:, :, :-1],
                    true_cat[:, 1:],
                    ignore_index=self.ignore_index,
                )
                ce_total += 1

            ce = ce / ce_total

        return mse + ce # TODO: Weights