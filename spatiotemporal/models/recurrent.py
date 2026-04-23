from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BatchGraphProjection, DiffusionGraphConv, diffusion_supports


class DCGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, adjacency: torch.Tensor, diffusion_steps: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        supports = diffusion_supports(adjacency)
        self.register_buffer("support_forward", supports[0])
        self.register_buffer("support_backward", supports[1])
        self.gate_conv = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim * 2, diffusion_steps)
        self.update_conv = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, diffusion_steps)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        supports = [self.support_forward, self.support_backward]
        gates = torch.sigmoid(self.gate_conv(torch.cat([x, hidden], dim=-1), supports))
        reset_gate, update_gate = torch.chunk(gates, 2, dim=-1)
        candidate = torch.tanh(self.update_conv(torch.cat([x, reset_gate * hidden], dim=-1), supports))
        return update_gate * hidden + (1.0 - update_gate) * candidate


class AGCRNCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_nodes: int,
        node_embedding_dim: int = 16,
        diffusion_steps: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, node_embedding_dim))
        self.gate_conv = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim * 2, diffusion_steps)
        self.update_conv = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, diffusion_steps)

    def _supports(self) -> list[torch.Tensor]:
        adjacency = F.softmax(F.relu(torch.matmul(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=-1)
        return [adjacency, adjacency.transpose(0, 1)]

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        supports = self._supports()
        gates = torch.sigmoid(self.gate_conv(torch.cat([x, hidden], dim=-1), supports))
        reset_gate, update_gate = torch.chunk(gates, 2, dim=-1)
        candidate = torch.tanh(self.update_conv(torch.cat([x, reset_gate * hidden], dim=-1), supports))
        return update_gate * hidden + (1.0 - update_gate) * candidate


class DGCRNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, diffusion_steps: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.graph_projection = BatchGraphProjection(input_dim + hidden_dim, hidden_dim)
        self.gate_conv = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim * 2, diffusion_steps)
        self.update_conv = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, diffusion_steps)

    def _supports(self, x: torch.Tensor, hidden: torch.Tensor) -> list[torch.Tensor]:
        combined = torch.cat([x, hidden], dim=-1)
        adjacency = self.graph_projection(combined)
        return [adjacency, adjacency.transpose(1, 2)]

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        supports = self._supports(x, hidden)
        gates = torch.sigmoid(self.gate_conv(torch.cat([x, hidden], dim=-1), supports))
        reset_gate, update_gate = torch.chunk(gates, 2, dim=-1)
        candidate = torch.tanh(self.update_conv(torch.cat([x, reset_gate * hidden], dim=-1), supports))
        return update_gate * hidden + (1.0 - update_gate) * candidate


class AutoregressiveGraphForecast(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        num_layers: int,
        cell_factory,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon_steps = horizon_steps
        self.num_layers = num_layers
        self.encoder_cells = nn.ModuleList()
        self.decoder_cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            self.encoder_cells.append(cell_factory(layer_input_dim, hidden_dim))
            self.decoder_cells.append(cell_factory(layer_input_dim, hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, 1)

    def _init_states(self, batch_size: int, num_nodes: int, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        return [
            torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]

    def _run_cells(self, cells: nn.ModuleList, inputs: torch.Tensor, states: list[torch.Tensor]):
        current = inputs
        next_states = []
        for layer, cell in enumerate(cells):
            next_hidden = cell(current, states[layer])
            next_states.append(next_hidden)
            current = next_hidden
        return current, next_states

    def _decoder_input(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.input_dim == prediction.size(-1):
            return prediction
        padding = prediction.new_zeros(prediction.size(0), prediction.size(1), self.input_dim - prediction.size(-1))
        return torch.cat([prediction, padding], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_nodes, _ = x.shape
        states = self._init_states(batch_size, num_nodes, x.device, x.dtype)
        for step in range(x.size(1)):
            _, states = self._run_cells(self.encoder_cells, x[:, step], states)

        decoder_states = states
        decoder_input = x[:, -1, :, :1]
        outputs = []
        for _ in range(self.horizon_steps):
            hidden, decoder_states = self._run_cells(
                self.decoder_cells,
                self._decoder_input(decoder_input),
                decoder_states,
            )
            decoder_input = self.output_proj(hidden)
            outputs.append(decoder_input)
        return torch.stack(outputs, dim=1)


class DCRNNModel(AutoregressiveGraphForecast):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        num_layers: int,
        adjacency: torch.Tensor,
        diffusion_steps: int = 2,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon_steps=horizon_steps,
            num_layers=num_layers,
            cell_factory=lambda layer_input_dim, layer_hidden_dim: DCGRUCell(
                layer_input_dim,
                layer_hidden_dim,
                adjacency=adjacency,
                diffusion_steps=diffusion_steps,
            ),
        )


class AGCRNModel(AutoregressiveGraphForecast):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        num_layers: int,
        num_nodes: int,
        node_embedding_dim: int = 16,
        diffusion_steps: int = 2,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon_steps=horizon_steps,
            num_layers=num_layers,
            cell_factory=lambda layer_input_dim, layer_hidden_dim: AGCRNCell(
                layer_input_dim,
                layer_hidden_dim,
                num_nodes=num_nodes,
                node_embedding_dim=node_embedding_dim,
                diffusion_steps=diffusion_steps,
            ),
        )


class DGCRNModel(AutoregressiveGraphForecast):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        num_layers: int,
        diffusion_steps: int = 2,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon_steps=horizon_steps,
            num_layers=num_layers,
            cell_factory=lambda layer_input_dim, layer_hidden_dim: DGCRNCell(
                layer_input_dim,
                layer_hidden_dim,
                diffusion_steps=diffusion_steps,
            ),
        )
