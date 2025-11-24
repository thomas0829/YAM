import math
import torch
from torch import Tensor, nn

class FixedSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """Learnable timestep embedding: sinusoid → MLP."""

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 4

        # fixed sinusoidal base
        self.sinu_emb = FixedSinusoidalPosEmb(dim)

        # learnable projection
        self.mlp = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.Mish(),
            nn.Linear(self.hidden_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B,) or (B, 1) timesteps (float or int)
        returns: (B, dim)
        """
        emb = self.sinu_emb(x)      # fixed features
        emb = self.mlp(emb)         # learned transform
        return emb
