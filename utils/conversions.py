import numpy as np
import torch


def tonp(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
