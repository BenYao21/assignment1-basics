import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device= None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device

        if d_k % 2 != 0:
            raise ValueError("d_k must be even.")

        # Precompute theta_i = theta^(-2i / d_k) for i in {0, 1, ..., d_k/2 - 1}
        theta_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        theta_vals = self.theta ** (-theta_indices / self.d_k)

        # Precompute position indices m
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)

        # Outer product to get m * theta_i
        m_theta = torch.outer(positions, theta_vals)

        cos_cached = torch.cos(m_theta)
        sin_cached = torch.sin(m_theta)

        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply Rotary Positional Embedding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Tensor of shape (..., seq_len)
                specifying the token positions.

        Returns:
            torch.Tensor: Output tensor of the same shape as x.
        """
        # Get cos and sin values for the given token positions
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # Split the input tensor into even and odd features
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply the rotation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Combine the rotated even and odd features
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated