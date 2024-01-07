import torch.nn as nn
import torch.optim as optim

from config import ConfigParams


def init_optimizer(config: ConfigParams, model: nn.Module):
    if config.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=config.learning_rate, momentum=config.momentum
        )
    elif config.optimizer == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise Exception("Missing optimizer")
    return optimizer
