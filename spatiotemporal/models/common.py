from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def row_normalize_adjacency(adjacency: torch.Tensor, add_self_loops: bool = True, eps: float = 1e-6) -> torch.Tensor:
    if add_self_loops:
        adjacency = adjacency + torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype)
    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(eps)
    return adjacency / degree


def symmetric_normalize_adjacency(adjacency: torch.Tensor, add_self_loops: bool = True, eps: float = 1e-6) -> torch.Tensor:
    if add_self_loops:
        adjacency = adjacency + torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype)
    degree = adjacency.sum(dim=-1).clamp_min(eps)
    degree_inv_sqrt = degree.pow(-0.5)
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]


def diffusion_supports(adjacency: torch.Tensor) -> list[torch.Tensor]:
    forward = row_normalize_adjacency(adjacency, add_self_loops=True)
    backward = row_normalize_adjacency(adjacency.transpose(0, 1), add_self_loops=True)
    return [forward, backward]


def propagate_nodes(x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    if support.dim() == 2:
        return torch.einsum("nm,bmc->bnc", support, x)
    if support.dim() == 3:
        return torch.einsum("bnm,bmc->bnc", support, x)
    raise ValueError(f"Unsupported support rank: {support.dim()}")


class DiffusionGraphConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        diffusion_steps: int = 2,
        num_supports: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.diffusion_steps = diffusion_steps
        self.num_supports = num_supports
        self.proj = nn.Linear(input_dim * (1 + num_supports * diffusion_steps), output_dim, bias=bias)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        outputs = [x]
        for support in supports:
            current = propagate_nodes(x, support)
            outputs.append(current)
            for _ in range(2, self.diffusion_steps + 1):
                current = propagate_nodes(current, support)
                outputs.append(current)
        return self.proj(torch.cat(outputs, dim=-1))


class ForecastHead(nn.Module):
    def __init__(self, input_dim: int, horizon_steps: int, output_dim: int = 1) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, horizon_steps * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.proj(x)
        batch_size, num_nodes, _ = output.shape
        return output.view(batch_size, num_nodes, self.horizon_steps, self.output_dim).permute(0, 2, 1, 3)


class ResidualMLP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, num_nodes, hidden_dim = x.shape
        residual = x
        x = self.norm(x)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, steps, hidden_dim)
        attended, _ = self.attn(x, x, x, need_weights=False)
        attended = attended.reshape(batch_size, num_nodes, steps, hidden_dim).permute(0, 2, 1, 3)
        return residual + self.dropout(attended)


class SpatialSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, num_nodes, hidden_dim = x.shape
        residual = x
        x = self.norm(x)
        x = x.reshape(batch_size * steps, num_nodes, hidden_dim)
        attended, _ = self.attn(x, x, x, need_weights=False)
        attended = attended.reshape(batch_size, steps, num_nodes, hidden_dim)
        return residual + self.dropout(attended)


class CrossTemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.key_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch_size, horizon_steps, num_nodes, hidden_dim = query.shape
        _, history_steps, _, _ = memory.permute(0, 1, 2, 3).shape
        query = self.query_norm(query).permute(0, 2, 1, 3).reshape(batch_size * num_nodes, horizon_steps, hidden_dim)
        memory = self.key_norm(memory).permute(0, 2, 1, 3).reshape(batch_size * num_nodes, history_steps, hidden_dim)
        attended, _ = self.attn(query, memory, memory, need_weights=False)
        attended = attended.reshape(batch_size, num_nodes, horizon_steps, hidden_dim).permute(0, 2, 1, 3)
        return self.dropout(attended)


class GraphConstructor(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int, top_k: int | None = None, alpha: float = 3.0) -> None:
        super().__init__()
        self.top_k = top_k
        self.alpha = alpha
        self.emb1 = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.emb2 = nn.Parameter(torch.randn(num_nodes, embedding_dim))

    def forward(self) -> torch.Tensor:
        logits = torch.matmul(self.emb1, self.emb2.transpose(0, 1)) - torch.matmul(
            self.emb2,
            self.emb1.transpose(0, 1),
        )
        adjacency = F.relu(torch.tanh(self.alpha * logits))
        if self.top_k is not None and 0 < self.top_k < adjacency.size(1):
            values, indices = torch.topk(adjacency, self.top_k, dim=-1)
            mask = torch.zeros_like(adjacency)
            mask.scatter_(1, indices, 1.0)
            adjacency = adjacency * mask
        return row_normalize_adjacency(adjacency, add_self_loops=True)


class DilatedInception(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dilation: int = 1,
        kernel_sizes: tuple[int, ...] = (2, 3, 6, 7),
    ) -> None:
        super().__init__()
        split_sizes = []
        remaining = output_dim
        for index in range(len(kernel_sizes)):
            channels = output_dim // len(kernel_sizes)
            if index < output_dim % len(kernel_sizes):
                channels += 1
            split_sizes.append(channels)
            remaining -= channels
        if remaining != 0:
            raise ValueError("Channel split mismatch in DilatedInception")

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    input_dim,
                    split_sizes[index],
                    kernel_size=(1, kernel_size),
                    dilation=(1, dilation),
                )
                for index, kernel_size in enumerate(kernel_sizes)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [conv(x) for conv in self.convs]
        min_length = min(output.size(-1) for output in outputs)
        outputs = [output[..., -min_length:] for output in outputs]
        return torch.cat(outputs, dim=1)


class MixProp(nn.Module):
    def __init__(self, channels: int, gdep: int = 2, dropout: float = 0.0, alpha: float = 0.05) -> None:
        super().__init__()
        self.gdep = gdep
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Conv2d((gdep + 1) * channels, channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        outputs = [x]
        hidden = x
        for _ in range(self.gdep):
            propagated = torch.einsum("nm,bcmt->bcnt", adjacency, hidden)
            hidden = self.alpha * x + (1 - self.alpha) * propagated
            outputs.append(hidden)
        mixed = self.mlp(torch.cat(outputs, dim=1))
        return self.dropout(mixed)


class BatchGraphProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query(x)
        key = self.key(x)
        scores = torch.einsum("bnc,bmc->bnm", query, key) / math.sqrt(query.size(-1))
        return F.softmax(F.relu(scores), dim=-1)
