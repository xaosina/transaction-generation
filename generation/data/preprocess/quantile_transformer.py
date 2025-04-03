import torch


class QuantileTransformerTorch:
    def __init__(self, n_quantiles=1000, output_distribution="uniform"):
        """
        :param n_quantiles: Quantile numbers
        :param output_distribution: 'uniform' or 'normal'
        """
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.quantiles_ = None
        self.references_ = None

    @staticmethod
    def _torch_interp(
        x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
    ) -> torch.Tensor:

        inds = torch.searchsorted(xp, x)
        inds = torch.clamp(inds, 1, len(xp) - 1)

        x0 = xp[inds - 1]
        x1 = xp[inds]
        y0 = fp[inds - 1]
        y1 = fp[inds]

        weight = (x - x0) / (x1 - x0)
        y = y0 + weight * (y1 - y0)

        y = torch.where(x < xp[0], fp[0].expand_as(y), y)
        y = torch.where(x > xp[-1], fp[-1].expand_as(y), y)
        return y

    def fit(self, amount_tensor: torch.Tensor):

        amount_sorted, _ = torch.sort(amount_tensor)
        n_samples = len(amount_sorted)
        quantile_indices = torch.linspace(0, n_samples - 1, self.n_quantiles).long()
        self.quantiles_ = amount_sorted[quantile_indices]
        assert torch.all(
            self.quantiles_[1:] >= self.quantiles_[:-1]
        ), "Quantile values should monotonically increase!"
        self.references_ = torch.linspace(0, 1, self.n_quantiles)

    def transform(self, batch: torch.Tensor) -> torch.Tensor:
        self.to(batch.device)
        u = self._torch_interp(batch, self.quantiles_, self.references_)
        if self.output_distribution == "uniform":
            return u
        elif self.output_distribution == "normal":
            eps = 1e-6
            u = u.clamp(eps, 1 - eps)
            z = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * u - 1)
            z = z.clamp(-10, 10)  # защита от экстремальных значений
            return z
        else:
            raise ValueError("Unknown target distribution: " + self.output_distribution)

    def inverse_transform(self, batch: torch.Tensor) -> torch.Tensor:
        self.to(batch.device)
        if self.output_distribution == "uniform":
            u = batch
        elif self.output_distribution == "normal":
            batch = batch.clamp(-10, 10)
            u = 0.5 * (1 + torch.erf(batch / torch.sqrt(torch.tensor(2.0))))
            eps = 1e-6
            u = u.clamp(eps, 1 - eps)
        else:
            raise ValueError("Unknown target distribution: " + self.output_distribution)
        return self._torch_interp(u, self.references_, self.quantiles_)

    def to(self, device):
        self.quantiles_ = self.quantiles_.to(device)
        self.references_ = self.references_.to(device)

    def save(self, path: str):
        torch.save(
            {
                "n_quantiles": self.n_quantiles,
                "output_distribution": self.output_distribution,
                "quantiles_": self.quantiles_,
                "references_": self.references_,
            },
            path,
        )

    def load(self, path: str):

        data = torch.load(path)
        self.n_quantiles = data["n_quantiles"]
        self.output_distribution = data["output_distribution"]
        self.quantiles_ = data["quantiles_"]
        self.references_ = data["references_"]
