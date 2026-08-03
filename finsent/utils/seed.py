"""
Reproducibility utilities.
Sets all random seeds and configures deterministic behaviour.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.
    
    Note: torch.backends.cudnn.deterministic = True slows training by ~10%
    but guarantees bitwise reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "cuda") -> torch.device:
    """Get compute device with graceful fallback."""
    if prefer == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Device] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device
