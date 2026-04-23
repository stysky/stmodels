from __future__ import annotations

import torch
import torch.nn as nn

from .common import CrossTemporalAttention, ResidualMLP, SpatialSelfAttention, TemporalSelfAttention


class GMANBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.temporal_attn = TemporalSelfAttention(hidden_dim, num_heads, dropout=dropout)
        self.spatial_attn = SpatialSelfAttention(hidden_dim, num_heads, dropout=dropout)
        self.feed_forward = ResidualMLP(hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_attn(x)
        x = self.spatial_attn(x)
        return self.feed_forward(x)


class GMANModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        history_steps: int,
        horizon_steps: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.history_embeddings = nn.Parameter(torch.randn(history_steps, hidden_dim))
        self.future_embeddings = nn.Parameter(torch.randn(horizon_steps, hidden_dim))
        self.history_blocks = nn.ModuleList(
            [GMANBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(num_blocks)]
        )
        self.cross_attention = CrossTemporalAttention(hidden_dim, num_heads, dropout=dropout)
        self.future_spatial = SpatialSelfAttention(hidden_dim, num_heads, dropout=dropout)
        self.future_ff = ResidualMLP(hidden_dim, dropout=dropout)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        node_bias = self.node_embeddings.unsqueeze(0).unsqueeze(0)
        history = self.input_proj(x)
        history = history + self.history_embeddings.unsqueeze(0).unsqueeze(2) + node_bias
        for block in self.history_blocks:
            history = block(history)

        future = self.future_embeddings.unsqueeze(0).unsqueeze(2) + node_bias
        future = future.expand(batch_size, -1, -1, -1)
        future = future + self.cross_attention(future, history)
        future = self.future_spatial(future)
        future = self.future_ff(future)
        return self.output_proj(future)


class STIDModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        history_steps: int,
        horizon_steps: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.history_proj = nn.Linear(history_steps * input_dim, hidden_dim)
        self.temporal_proj = nn.Linear(input_dim, hidden_dim)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.blocks = nn.ModuleList([ResidualMLP(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.output_proj = nn.Linear(hidden_dim, horizon_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, history_steps, num_nodes, input_dim = x.shape
        history = x.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, history_steps * input_dim)
        hidden = self.history_proj(history)
        hidden = hidden + self.temporal_proj(x[:, -1]) + self.node_embeddings.unsqueeze(0)
        for block in self.blocks:
            hidden = block(hidden)
        output = self.output_proj(hidden)
        return output.permute(0, 2, 1).unsqueeze(-1)
