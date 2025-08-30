import torch
from typing import Iterable
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in params_with_grad))
    if norm > max_l2_norm:
        for p in params_with_grad:
            p.grad.data *= max_l2_norm / (norm + eps)