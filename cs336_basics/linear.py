import torch
import einops

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: str = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.randn(out_features, in_features, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(in_features * out_features, out_features, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flattened = einops.rearrange(x, "b ... -> b (...)")
        output = self.linear(x_flattened)
        return output