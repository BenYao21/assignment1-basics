import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .multihead_self_attention import MultiheadSelfAttention
from .positionwise_feedforward import SwiGLU
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # MultiheadSelfAttention 需要在初始化时传入权重。
        # 但 TransformerBlock 是先初始化，再通过 load_state_dict 加载权重。
        # 因此，我们在这里创建虚拟（dummy）权重，它们稍后会被真实权重覆盖。
        dummy_q = torch.empty((d_model, d_model), device=device, dtype=dtype)
        dummy_k = torch.empty((d_model, d_model), device=device, dtype=dtype)
        dummy_v = torch.empty((d_model, d_model), device=device, dtype=dtype)
        dummy_o = torch.empty((d_model, d_model), device=device, dtype=dtype)

        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            q_proj_weight=dummy_q,
            k_proj_weight=dummy_k,
            v_proj_weight=dummy_v,
            o_proj_weight=dummy_o,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def load_state_dict(self, state_dict, strict=True):
        # The test suite provides a state_dict with the key "attn.output_proj.weight",
        # but our MultiheadSelfAttention class names this layer "o_proj".
        # To resolve this mismatch, we rename the key in the state_dict before loading.
        if "attn.output_proj.weight" in state_dict:
            state_dict["attn.o_proj.weight"] = state_dict.pop("attn.output_proj.weight")
        return super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        x: Float[Tensor, "batch sequence_length d_model"],
        token_positions=None,
    ) -> Float[Tensor, "batch sequence_length d_model"]:
        # Pre-norm 结构
        if token_positions is None:
            # If token positions are not provided, we assume the input is a
            # contiguous sequence of tokens.
            seq_len = x.shape[1]
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x