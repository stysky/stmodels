from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..preprocessing import graph_wavenet_supports
from .common import DiffusionGraphConv, DilatedInception, ForecastHead, GraphConstructor, MixProp


class _GraphWaveNetNodeConv(nn.Module):
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcnt,nm->bcmt", x, adjacency).contiguous()


class _GraphWaveNetLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(input_dim, output_dim, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _GraphWaveNetGCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float, support_len: int = 3, order: int = 2) -> None:
        super().__init__()
        self.node_conv = _GraphWaveNetNodeConv()
        self.order = order
        expanded_dim = (order * support_len + 1) * input_dim
        self.linear = _GraphWaveNetLinear(expanded_dim, output_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        outputs = [x]
        for adjacency in supports:
            current = self.node_conv(x, adjacency)
            outputs.append(current)
            for _ in range(2, self.order + 1):
                current = self.node_conv(current, adjacency)
                outputs.append(current)
        hidden = self.linear(torch.cat(outputs, dim=1))
        return F.dropout(hidden, self.dropout, training=self.training)


class _GraphWaveNetCore(nn.Module):
    def __init__(
        self,
        device: torch.device,
        num_nodes: int,
        dropout: float = 0.3,
        supports: list[torch.Tensor] | None = None,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        aptinit: torch.Tensor | None = None,
        in_dim: int = 2,
        out_dim: int = 12,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        self.supports = supports

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.graph_convs = nn.ModuleList()

        receptive_field = 1
        self.supports_len = 0 if supports is None else len(supports)
        if gcn_bool and addaptadj:
            if supports is None:
                self.supports = []
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10, device=device))
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes, device=device))
            else:
                m, p, n = torch.svd(aptinit)
                self.nodevec1 = nn.Parameter(torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5)).to(device))
                self.nodevec2 = nn.Parameter(torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t()).to(device))
            self.supports_len += 1

        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, new_dilation),
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, new_dilation),
                    )
                )
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.batch_norms.append(nn.BatchNorm2d(residual_channels))
                if self.gcn_bool:
                    self.graph_convs.append(
                        _GraphWaveNetGCN(
                            dilation_channels,
                            residual_channels,
                            dropout,
                            support_len=self.supports_len,
                        )
                    )
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))
        x = self.start_conv(x)
        skip = 0

        adaptive_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adaptive_adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adaptive_supports = list(self.supports) + [adaptive_adj]

        for layer_idx in range(self.blocks * self.layers):
            residual = x
            filter_out = torch.tanh(self.filter_convs[layer_idx](residual))
            gate_out = torch.sigmoid(self.gate_convs[layer_idx](residual))
            x = filter_out * gate_out

            skip_out = self.skip_convs[layer_idx](x)
            try:
                skip = skip[:, :, :, -skip_out.size(3) :]
            except Exception:
                skip = 0
            skip = skip_out + skip

            if self.gcn_bool and self.supports is not None:
                supports = adaptive_supports if self.addaptadj else self.supports
                x = self.graph_convs[layer_idx](x, supports)
            else:
                x = self.residual_convs[layer_idx](x)

            x = x + residual[:, :, :, -x.size(3) :]
            x = self.batch_norms[layer_idx](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        return self.end_conv_2(x)


class _STGCNAlign(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.align_conv = nn.Conv2d(input_dim, output_dim, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_dim > self.output_dim:
            return self.align_conv(x)
        if self.input_dim < self.output_dim:
            batch_size, _, steps, num_nodes = x.shape
            zeros = x.new_zeros(batch_size, self.output_dim - self.input_dim, steps, num_nodes)
            return torch.cat([x, zeros], dim=1)
        return x


class _STGCNCausalConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        enable_padding: bool = False,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding:
            self._padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self._padding = 0
        self.left_padding = nn.modules.utils._pair(self._padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._padding != 0:
            x = F.pad(x, (self.left_padding[1], 0, self.left_padding[0], 0))
        return super().forward(x)


class _STGCNTemporalConvLayer(nn.Module):
    def __init__(self, kernel_size: int, input_dim: int, output_dim: int, act_func: str) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.align = _STGCNAlign(input_dim, output_dim)
        if act_func in {"glu", "gtu"}:
            conv_out = 2 * output_dim
        else:
            conv_out = output_dim
        self.causal_conv = _STGCNCausalConv2d(
            in_channels=input_dim,
            out_channels=conv_out,
            kernel_size=(kernel_size, 1),
            enable_padding=False,
        )
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.act_func = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aligned = self.align(x)[:, :, self.kernel_size - 1 :, :]
        causal_out = self.causal_conv(x)
        if self.act_func in {"glu", "gtu"}:
            primary = causal_out[:, : self.output_dim, :, :]
            gate = causal_out[:, -self.output_dim :, :, :]
            if self.act_func == "glu":
                return torch.mul(primary + aligned, torch.sigmoid(gate))
            return torch.mul(torch.tanh(primary + aligned), torch.sigmoid(gate))
        if self.act_func == "relu":
            return self.relu(causal_out + aligned)
        if self.act_func == "silu":
            return self.silu(causal_out + aligned)
        raise NotImplementedError(f"Unsupported STGCN activation: {self.act_func}")


class _STGCNChebGraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, order: int, gso: torch.Tensor, bias: bool) -> None:
        super().__init__()
        self.order = order
        self.gso = gso
        self.weight = nn.Parameter(torch.empty(order, input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        if self.order <= 0:
            raise ValueError(f"STGCN graph kernel size must be positive, got {self.order}")
        basis = [x]
        if self.order > 1:
            basis.append(torch.einsum("hi,btij->bthj", self.gso, x))
            for _ in range(2, self.order):
                basis.append(torch.einsum("hi,btij->bthj", 2 * self.gso, basis[-1]) - basis[-2])
        stacked = torch.stack(basis, dim=2)
        output = torch.einsum("btkhi,kij->bthj", stacked, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output


class _STGCNGraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, gso: torch.Tensor, bias: bool) -> None:
        super().__init__()
        self.gso = gso
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        first_mul = torch.einsum("hi,btij->bthj", self.gso, x)
        output = torch.einsum("bthi,ij->bthj", first_mul, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output


class _STGCNGraphConvLayer(nn.Module):
    def __init__(
        self,
        graph_conv_type: str,
        input_dim: int,
        output_dim: int,
        order: int,
        gso: torch.Tensor,
        bias: bool,
    ) -> None:
        super().__init__()
        self.graph_conv_type = graph_conv_type
        self.align = _STGCNAlign(input_dim, output_dim)
        if graph_conv_type == "cheb_graph_conv":
            self.graph_conv = _STGCNChebGraphConv(output_dim, output_dim, order, gso, bias)
        elif graph_conv_type == "graph_conv":
            self.graph_conv = _STGCNGraphConv(output_dim, output_dim, gso, bias)
        else:
            raise ValueError(f"Unsupported STGCN graph conv type: {graph_conv_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aligned = self.align(x)
        graph_out = self.graph_conv(aligned).permute(0, 3, 1, 2)
        return graph_out + aligned


class _STGCNBlock(nn.Module):
    def __init__(
        self,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        num_nodes: int,
        input_dim: int,
        channels: list[int],
        act_func: str,
        graph_conv_type: str,
        gso: torch.Tensor,
        bias: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temp_conv1 = _STGCNTemporalConvLayer(temporal_kernel_size, input_dim, channels[0], act_func)
        self.graph_conv = _STGCNGraphConvLayer(
            graph_conv_type,
            channels[0],
            channels[1],
            spatial_kernel_size,
            gso,
            bias,
        )
        self.temp_conv2 = _STGCNTemporalConvLayer(temporal_kernel_size, channels[1], channels[2], act_func)
        self.layer_norm = nn.LayerNorm([num_nodes, channels[2]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.temp_conv2(x)
        x = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)


class _STGCNOutputBlock(nn.Module):
    def __init__(
        self,
        output_kernel_size: int,
        input_dim: int,
        channels: list[int],
        end_channel: int,
        num_nodes: int,
        act_func: str,
        bias: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temp_conv1 = _STGCNTemporalConvLayer(output_kernel_size, input_dim, channels[0], act_func)
        self.layer_norm = nn.LayerNorm([num_nodes, channels[0]], eps=1e-12)
        self.fc1 = nn.Linear(channels[0], channels[1], bias=bias)
        self.fc2 = nn.Linear(channels[1], end_channel, bias=bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temp_conv1(x)
        x = self.layer_norm(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x).permute(0, 3, 1, 2)


class _STGCNCore(nn.Module):
    def __init__(
        self,
        history_steps: int,
        blocks: list[list[int]],
        num_nodes: int,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        act_func: str,
        graph_conv_type: str,
        gso: torch.Tensor,
        enable_bias: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        modules = []
        for index in range(len(blocks) - 3):
            modules.append(
                _STGCNBlock(
                    temporal_kernel_size,
                    spatial_kernel_size,
                    num_nodes,
                    blocks[index][-1],
                    blocks[index + 1],
                    act_func,
                    graph_conv_type,
                    gso,
                    enable_bias,
                    dropout,
                )
            )
        self.st_blocks = nn.Sequential(*modules)
        output_kernel_size = history_steps - (len(blocks) - 3) * 2 * (temporal_kernel_size - 1)
        self.output_kernel_size = output_kernel_size
        if output_kernel_size > 1:
            self.output_block = _STGCNOutputBlock(
                output_kernel_size,
                blocks[-3][-1],
                blocks[-2],
                blocks[-1][0],
                num_nodes,
                act_func,
                enable_bias,
                dropout,
            )
        elif output_kernel_size == 0:
            self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0], bias=enable_bias)
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0], bias=enable_bias)
            self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.st_blocks(x)
        if self.output_kernel_size > 1:
            return self.output_block(x)
        if self.output_kernel_size == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            return self.fc2(x).permute(0, 3, 1, 2)
        return x


class GraphWaveNetModel(nn.Module):
    def __init__(
        self,
        project_root: str | Path,
        input_dim: int,
        num_nodes: int,
        horizon_steps: int,
        adjacency: torch.Tensor,
        dropout: float = 0.3,
        nhid: int = 32,
        adjtype: str = "doubletransition",
        gcn_bool: bool = True,
        addaptadj: bool = True,
        aptonly: bool = False,
        randomadj: bool = False,
    ) -> None:
        super().__init__()
        supports = [
            torch.tensor(arr, dtype=torch.float32, device=adjacency.device)
            for arr in graph_wavenet_supports(adjacency.detach().cpu().numpy(), adjtype)
        ]
        adjinit = None if randomadj or not supports else supports[0]
        if aptonly:
            supports = None

        self.model = _GraphWaveNetCore(
            device=adjacency.device,
            num_nodes=num_nodes,
            dropout=dropout,
            supports=supports,
            gcn_bool=gcn_bool,
            addaptadj=addaptadj,
            aptinit=adjinit,
            in_dim=input_dim,
            out_dim=horizon_steps,
            residual_channels=nhid,
            dilation_channels=nhid,
            skip_channels=nhid * 8,
            end_channels=nhid * 16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model_input = F.pad(x.permute(0, 3, 2, 1), (1, 0, 0, 0))
        return self.model(model_input)


class STGCNModel(nn.Module):
    def __init__(
        self,
        project_root: str | Path,
        input_dim: int,
        num_nodes: int,
        history_steps: int,
        horizon_steps: int,
        gso: torch.Tensor,
        Kt: int = 3,
        Ks: int = 3,
        stblock_num: int = 2,
        act_func: str = "glu",
        graph_conv_type: str = "cheb_graph_conv",
        enable_bias: bool = True,
        droprate: float = 0.5,
    ) -> None:
        super().__init__()
        ko = history_steps - (Kt - 1) * 2 * stblock_num
        if ko < 0:
            raise ValueError("STGCN history_steps is too small for the configured temporal kernel and block count")

        blocks = [[input_dim]]
        for _ in range(stblock_num):
            blocks.append([64, 16, 64])
        if ko == 0:
            blocks.append([128])
        else:
            blocks.append([128, 128])
        blocks.append([horizon_steps])

        self.model = _STGCNCore(
            history_steps=history_steps,
            blocks=blocks,
            num_nodes=num_nodes,
            temporal_kernel_size=Kt,
            spatial_kernel_size=Ks,
            act_func=act_func,
            graph_conv_type=graph_conv_type,
            gso=gso,
            enable_bias=enable_bias,
            dropout=droprate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x.permute(0, 3, 1, 2))
        return output.permute(0, 1, 3, 2)


class ASTGCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, diffusion_steps: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.temporal_query = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_key = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_query = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_key = nn.Linear(hidden_dim, hidden_dim)
        self.graph_conv = DiffusionGraphConv(
            hidden_dim,
            hidden_dim,
            diffusion_steps=diffusion_steps,
            num_supports=1,
        )
        self.temporal_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(x.size(-1))
        temporal_query = self.temporal_query(x.mean(dim=2))
        temporal_key = self.temporal_key(x.mean(dim=2))
        temporal_scores = torch.softmax(
            torch.einsum("btc,bsc->bts", temporal_query, temporal_key) / scale,
            dim=-1,
        )
        attended_time = torch.einsum("bts,bsnc->btnc", temporal_scores, x)

        spatial_query = self.spatial_query(attended_time.mean(dim=1))
        spatial_key = self.spatial_key(attended_time.mean(dim=1))
        spatial_scores = torch.softmax(
            torch.einsum("bnc,bmc->bnm", spatial_query, spatial_key) / scale,
            dim=-1,
        )
        support = adjacency.unsqueeze(0) * spatial_scores
        identity = torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
        support = support + identity
        support = support / support.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        graph_outputs = []
        for step in range(attended_time.size(1)):
            graph_outputs.append(self.graph_conv(attended_time[:, step], [support]))
        graph_out = torch.stack(graph_outputs, dim=1)

        temporal_out = self.temporal_conv(graph_out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        residual = self.residual_proj(x)
        return self.dropout(F.relu(self.norm(temporal_out + residual)))


class ASTGCNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon_steps: int,
        adjacency: torch.Tensor,
        num_blocks: int = 2,
        diffusion_steps: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ASTGCNBlock(hidden_dim, diffusion_steps=diffusion_steps, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head = ForecastHead(hidden_dim, horizon_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, self.adjacency)
        summary = 0.5 * x[:, -1] + 0.5 * x.mean(dim=1)
        return self.head(summary)


class MTGNNBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float = 0.1, gdep: int = 2) -> None:
        super().__init__()
        self.filter_conv = DilatedInception(hidden_dim, hidden_dim, dilation=dilation, kernel_sizes=(2, 3))
        self.gate_conv = DilatedInception(hidden_dim, hidden_dim, dilation=dilation, kernel_sizes=(2, 3))
        self.residual_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.skip_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.mixprop = MixProp(hidden_dim, gdep=gdep, dropout=dropout)
        self.norm = nn.BatchNorm2d(hidden_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor):
        residual = x
        x = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        graph_out = self.mixprop(x, adjacency)
        x = graph_out + self.residual_conv(x)
        x = x + residual[..., -x.size(-1) :]
        x = self.norm(x)
        skip = self.skip_conv(x)
        return x, skip


class MTGNNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        horizon_steps: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        top_k: int = 20,
        dropout: float = 0.1,
        gdep: int = 2,
    ) -> None:
        super().__init__()
        self.graph_constructor = GraphConstructor(num_nodes, hidden_dim, top_k=top_k)
        self.start_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
        self.blocks = nn.ModuleList(
            [MTGNNBlock(hidden_dim, dilation=2**layer, dropout=dropout, gdep=gdep) for layer in range(num_layers)]
        )
        self.end_conv_1 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(hidden_dim * 2, horizon_steps, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = self.graph_constructor()
        x = self.start_conv(x.permute(0, 3, 2, 1))
        skip = None
        for block in self.blocks:
            x, block_skip = block(x, adjacency)
            skip = block_skip if skip is None else skip[..., -block_skip.size(-1) :] + block_skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)[..., -1:]
        return x
