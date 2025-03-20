import torch
from torch import nn
from torch.distributions import Categorical


class ProjectMSE:
    def __init__(self, d_model, cfg):
        self.projector = nn.Linear(d_model, cfg.hidden_dim)

    def forward(self, x):
        return self.projector(x)

    def generate(self, seq):
        return torch.cat([self.projector(x) for x in seq])


class ProjectCE:
    def __init__(self, d_model, cfg):
        self.c_dim = cfg.codes_dim
        self.c_number = cfg.codes_number
        self.last_linear = nn.Linear(d_model, 2 * d_model)
        self.classifier = nn.Linear(2 * d_model, cfg.codes_dim * cfg.codes_number)

    def forward(self, x):
        x = self.last_linear(x)
        x = self.classifier(x)
        return x  # TODO: Permute x dims for cross entropy

    def generate(self, seq, sampling_temperature=1.0):
        res = []
        for x in seq:
            x = self.last_linear(x)
            logits = self.classifier(x)
            x = Categorical(
                logits=logits.view(-1, self.c_number, self.c_dim) / sampling_temperature
            ).sample()
            res.append(x)
        return torch.cat(res)
