import torch
import einops

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: str = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 按照要求，创建一个空的 Parameter，形状为 (out_features, in_features)
        # 具体的数值将由 reset_parameters 方法填充
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        # 调用初始化方法
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 核心矩阵乘法 Y = XW^T，使用 einops 实现
        # 注意：这里不再有偏置项 + b
        output = einops.einsum(
            x, self.weight, "... in_features, out_features in_features -> ... out_features"
        )
        return output