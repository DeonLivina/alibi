import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (both CPU and CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Make PyTorch operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For reproducibility across hash-based operations (Python 3.3+)
    os.environ["PYTHONHASHSEED"] = str(seed)
