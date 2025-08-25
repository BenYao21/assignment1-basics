import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .rope import RotaryPositionalEmbedding
from .scaled_dot_product_attention import scaled_dot_product_attention


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, "d_model d_model"],
        k_proj_weight: Float[Tensor, "d_model d_model"],
        v_proj_weight: Float[Tensor, "d_model d_model"],
        o_proj_weight: Float[Tensor, "d_model d_model"],
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_proj.weight.data.copy_(q_proj_weight)
        self.k_proj.weight.data.copy_(k_proj_weight)
        self.v_proj.weight.data.copy_(v_proj_weight)
        self.o_proj.weight.data.copy_(o_proj_weight)

        self.rope = None
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta, d_k=self.d_head, max_seq_len=max_seq_len
            )

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
        token_positions: Int[Tensor, "... sequence_length"] | None = None,
    ) -> Tensor:
        batch_dims = x.shape[:-2]
        seq_len = x.shape[-2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(*batch_dims, seq_len, self.num_heads, self.d_head).transpose(-3, -2)
        k = k.view(*batch_dims, seq_len, self.num_heads, self.d_head).transpose(-3, -2)
        v = v.view(*batch_dims, seq_len, self.num_heads, self.d_head).transpose(-3, -2)

        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)


        #casual_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        # Create a mask where True indicates positions that should be masked out (future positions)
        future_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        # Invert the mask so that True indicates positions to attend to
        causal_mask = ~future_mask

        attention_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        attention_output = (
            attention_output.transpose(-3, -2)
            .contiguous()
            .view(*batch_dims, seq_len, self.d_model)
        )

        return self.o_proj(attention_output)