from torch import nn
from torch.distributions import Categorical

class ProjectMSE:
    def __init__(self, d_model, cfg):
        self.classifier = nn.Linear(d_model, cfg.hidden_dim)

    def forward(self, x):
        return self.classifier(x)
    
    def generate(self, x):
        x = self.classifier(x)
        return x[:, -1]
    
class ProjectCE:
    def __init__(self, d_model, cfg):
        self.last_linear = nn.Linear(d_model, 2*d_model)
        self.classifier = nn.Linear(2*d_model, cfg.codes_dim * cfg.codes_number)

    def forward(self, x):
        x = self.last_linear(x)
        x = self.classifier(x)
        return x # TODO: Permute x dims for cross entropy
    
    def generate(self, x, sampling_temperature=1.0):
        x = self.last_linear(x)
        x = self.classifier(x)
        logits = x[:, -1, :]
        x = Categorical(logits=logits.view(-1, self.c_number, self.c_dim) / sampling_temperature).sample()
        return x
