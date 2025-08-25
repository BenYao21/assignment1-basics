import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes a numerically stable softmax.

    Args:
        x: Input tensor.
        dim: Dimension to apply softmax over.

    Returns:
        Result of softmax.
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    partition = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / partition
