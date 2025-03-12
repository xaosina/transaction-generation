from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    """ Training config for Machine Learning """
    # The number of workers for training
    workers: int = field(default=8) # The number of aaaaworkers for training
    # The number of aaaaworkers for training
    # The experiment name
    exp_name: str = field(default='default_exp')


def train(cfg: TrainConfig):
    pass