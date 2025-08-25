import torch

from .softmax import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    p_attn = softmax(scores, dim=-1)
    return torch.matmul(p_attn, V)