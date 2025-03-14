import torch
import torch.nn.functional as F


class BaseLoss:

    def __init__(self, loss_type="full"):
        self.loss_type = loss_type
        self._ignore_index = -100

    def _shift_tensor(self, tensor: torch.Tensor):
        tensor = tensor.roll(-1, dims=1)
        tensor[:, -1] = self._ignore_index
        return tensor

    def validate_target_list(self, targets: torch.Tensor):
        if not isinstance(targets, torch.Tensor):
            raise ValueError("Targets should be torch.Tensor")

    def _compute_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        self.validate_target_list(y_true)

        match self.loss_type:
            case "full":
                return self._autoregressive_loss(y_true, y_pred)
            case "last":
                return self._last_token_loss(y_true, y_pred)
            case _:
                raise ValueError(f"Неизвестный тип лосса: {self.loss_type}")

    def _autoregressive_loss(self, *args, **kwargs):
        raise NotImplementedError()
    
    def _last_token_loss(self, *args, **kwargs):
        raise NotImplementedError()
    
    def __call__(self, *args, **kwargs):
        return self._compute_loss(*args, **kwargs)


class CatLoss(BaseLoss):

    def __init__(self, c_dim, c_number, loss_type="full"):
        super().__init__(loss_type)

        self.c_dim = c_dim
        self.c_number = c_number

    def __call__(self, *args, **kwargs):
        return self._compute_loss(*args, **kwargs)

    def _autoregressive_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = self._shift_tensor(y_true)
        return F.cross_entropy(
            y_pred[:, :-1, ...].reshape(-1, self.c_number, self.c_dim).permute(0, 2, 1), 
            y_true[:, :-1, ...].reshape(-1, self.c_number).long()
            )
    
    def _last_token_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        return F.cross_entropy(
            y_pred[:, -1, ...].reshape(-1, self.c_number, self.c_dim).permute(0, 2, 1), 
            y_true[:, ...].reshape(-1, self.c_number).long()
            )


class MSELoss(BaseLoss):

    def __init__(self, loss_type="full"):
        super().__init__(loss_type)

    def __call__(self, *args, **kwargs):
        return self._compute_loss(*args, **kwargs)

    def _autoregressive_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = self._shift_tensor(y_true)
        return F.mse_loss(
            y_pred[:, :-1, :],
            y_true[:, :-1, :]
        )

    def _last_token_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        return F.mse_loss(y_pred[:, -1, :], y_true[:, -1, :])