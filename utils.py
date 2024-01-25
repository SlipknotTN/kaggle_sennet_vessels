import os
import random

import numpy as np
import torch


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device


def set_seed(seed: int):
    """
    Sets the seed to reproduce the same results for every run and better compare local
    with cloud notebook
    From: https://www.kaggle.com/code/vinayaktiwari28/easy-to-understand-clean-baseline-code-train
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
