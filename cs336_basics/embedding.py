from torch.nn import Embedding, Module
import torch

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: str = None, dtype = None):
        super().__init__()
        self.embedding = Embedding(num_embeddings, embedding_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)