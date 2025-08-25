import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .linear import Linear
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(self, 
        d_model: int,
        vocab_size: int, 
        num_layers: int, 
        num_heads: int, 
        d_ff: int, 
        max_seq_len: int, 
        theta: float,
        device=None,
        dtype=None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.embeddings = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, x: Int[Tensor, "batch sequence_length"]):
        x = self.embeddings(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key == "token_embeddings.weight":
                new_key = "embeddings.weight"
            elif new_key.startswith("layers."):
                new_key = new_key.replace("layers", "transformer_blocks", 1)

            if "attn.output_proj.weight" in new_key:
                new_key = new_key.replace("attn.output_proj.weight", "attn.o_proj.weight")
            
            new_state_dict[new_key] = value
        
        return super().load_state_dict(new_state_dict, strict=strict)