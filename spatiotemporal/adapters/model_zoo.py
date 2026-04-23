from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ..models import (
    AGCRNModel,
    ASTGCNModel,
    DCRNNModel,
    DGCRNModel,
    GMANModel,
    GraphWaveNetModel,
    MTGNNModel,
    STGCNModel,
    STIDModel,
)
from ..preprocessing import calc_chebynet_gso, calc_gso
from .unified import ForecastingBundle, ForecastingConfig, UnifiedForecastAdapter

ModelBuilder = Callable[[UnifiedForecastAdapter, ForecastingBundle, torch.device], torch.nn.Module]
SchedulerBuilder = Callable[[UnifiedForecastAdapter, Any], Any]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    adapter_cls_name: str
    config_cls: type[ForecastingConfig]
    build_model: ModelBuilder
    build_scheduler: SchedulerBuilder | None = None


class RegisteredForecastAdapter(UnifiedForecastAdapter):
    spec: ModelSpec
    model_name = ""
    config_cls = ForecastingConfig

    def _build_model(self, bundle: ForecastingBundle, device: torch.device):
        return self.spec.build_model(self, bundle, device)

    def _build_scheduler(self, optimizer):
        if self.spec.build_scheduler is None:
            return super()._build_scheduler(optimizer)
        return self.spec.build_scheduler(self, optimizer)


@dataclass
class GraphWaveNetConfig(ForecastingConfig):
    batch_size: int = 64
    epochs: int = 20
    dropout: float = 0.3
    nhid: int = 32
    adjtype: str = "doubletransition"
    gcn_bool: bool = True
    addaptadj: bool = True
    aptonly: bool = False
    randomadj: bool = False


@dataclass
class STGCNConfig(ForecastingConfig):
    batch_size: int = 32
    weight_decay: float = 0.001
    step_size: int = 10
    gamma: float = 0.95
    Kt: int = 3
    Ks: int = 3
    stblock_num: int = 2
    act_func: str = "glu"
    graph_conv_type: str = "cheb_graph_conv"
    gso_type: str = "sym_norm_lap"
    enable_bias: bool = True
    droprate: float = 0.5


@dataclass
class DCRNNConfig(ForecastingConfig):
    hidden_dim: int = 64
    num_layers: int = 2
    diffusion_steps: int = 2


@dataclass
class AGCRNConfig(ForecastingConfig):
    hidden_dim: int = 64
    num_layers: int = 2
    diffusion_steps: int = 2
    node_embedding_dim: int = 16


@dataclass
class ASTGCNConfig(ForecastingConfig):
    hidden_dim: int = 64
    num_blocks: int = 2
    diffusion_steps: int = 1
    dropout: float = 0.1


@dataclass
class GMANConfig(ForecastingConfig):
    batch_size: int = 32
    hidden_dim: int = 64
    num_heads: int = 4
    num_blocks: int = 2
    dropout: float = 0.1


@dataclass
class MTGNNConfig(ForecastingConfig):
    hidden_dim: int = 32
    num_layers: int = 2
    top_k: int = 20
    dropout: float = 0.1
    gdep: int = 2


@dataclass
class STIDConfig(ForecastingConfig):
    batch_size: int = 128
    weight_decay: float = 0.0
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1


@dataclass
class DGCRNConfig(ForecastingConfig):
    hidden_dim: int = 64
    num_layers: int = 2
    diffusion_steps: int = 2


def build_graph_wavenet(adapter: UnifiedForecastAdapter, bundle: ForecastingBundle, device: torch.device):
    config = adapter.config
    return GraphWaveNetModel(
        project_root=adapter.project_root,
        input_dim=bundle.input_dim,
        num_nodes=bundle.data.num_nodes,
        horizon_steps=config.horizon_steps,
        adjacency=adapter.adjacency_tensor(bundle, device),
        dropout=config.dropout,
        nhid=config.nhid,
        adjtype=config.adjtype,
        gcn_bool=config.gcn_bool,
        addaptadj=config.addaptadj,
        aptonly=config.aptonly,
        randomadj=config.randomadj,
    ).to(device)


def build_stgcn(adapter: UnifiedForecastAdapter, bundle: ForecastingBundle, device: torch.device):
    config = adapter.config
    gso = calc_gso(bundle.adjacency, config.gso_type)
    if config.graph_conv_type == "cheb_graph_conv":
        gso = calc_chebynet_gso(gso)
    gso_tensor = torch.from_numpy(np.asarray(gso.toarray(), dtype=np.float32)).to(device)
    return STGCNModel(
        project_root=adapter.project_root,
        input_dim=bundle.input_dim,
        num_nodes=bundle.data.num_nodes,
        history_steps=config.history_steps,
        horizon_steps=config.horizon_steps,
        gso=gso_tensor,
        Kt=config.Kt,
        Ks=config.Ks,
        stblock_num=config.stblock_num,
        act_func=config.act_func,
        graph_conv_type=config.graph_conv_type,
        enable_bias=config.enable_bias,
        droprate=config.droprate,
    ).to(device)


def build_stgcn_scheduler(adapter: UnifiedForecastAdapter, optimizer):
    config = adapter.config
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.step_size,
        gamma=config.gamma,
    )


def build_dcrnn(adapter: UnifiedForecastAdapter, bundle: ForecastingBundle, device: torch.device):
    config = adapter.config
    return DCRNNModel(
        input_dim=bundle.input_dim,
        hidden_dim=config.hidden_dim,
        horizon_steps=config.horizon_steps,
        num_layers=config.num_layers,
        adjacency=adapter.adjacency_tensor(bundle, device),
        diffusion_steps=config.diffusion_steps,
    ).to(device)


def build_agcrn(adapter: UnifiedForecastAdapter, bundle: ForecastingBundle, device: torch.device):
    config = adapter.config
    return AGCRNModel(
        input_dim=bundle.input_dim,
        hidden_dim=config.hidden_dim,
        horizon_steps=config.horizon_steps,
        num_layers=config.num_layers,
        num_nodes=bundle.data.num_nodes,
        node_embedding_dim=config.node_embedding_dim,
        diffusion_steps=config.diffusion_steps,
    ).to(device)


def build_astgcn(adapter: UnifiedForecastAdapter, bundle: ForecastingBundle, device: torch.device):
    config = adapter.config
    return ASTGCNModel(
        input_dim=bundle.input_dim,
        hidden_dim=config.hidden_dim,
        horizon_steps=config.horizon_steps,
        adjacency=adapter.adjacency_tensor(bundle, device),
        num_blocks=config.num_blocks,
        diffusion_steps=config.diffusion_steps,
        dropout=config.dropout,
    ).to(device)


def build_gman(adapter: UnifiedForecastAdapter, bundle: ForecastingBundle, device: torch.device):
    config = adapter.config
    return GMANModel(
        input_dim=bundle.input_dim,
        num_nodes=bundle.data.num_nodes,
        history_steps=config.history_steps,
        horizon_steps=config.horizon_steps,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
    ).to(device)


def build_mtgnn(adapter: UnifiedForecastAdapter, bundle: ForecastingBundle, device: torch.device):
    config = adapter.config
    return MTGNNModel(
        input_dim=bundle.input_dim,
        num_nodes=bundle.data.num_nodes,
        horizon_steps=config.horizon_steps,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        top_k=config.top_k,
        dropout=config.dropout,
        gdep=config.gdep,
    ).to(device)


def build_stid(adapter: UnifiedForecastAdapter, bundle: ForecastingBundle, device: torch.device):
    config = adapter.config
    return STIDModel(
        input_dim=bundle.input_dim,
        num_nodes=bundle.data.num_nodes,
        history_steps=config.history_steps,
        horizon_steps=config.horizon_steps,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)


def build_dgcrn(adapter: UnifiedForecastAdapter, bundle: ForecastingBundle, device: torch.device):
    config = adapter.config
    return DGCRNModel(
        input_dim=bundle.input_dim,
        hidden_dim=config.hidden_dim,
        horizon_steps=config.horizon_steps,
        num_layers=config.num_layers,
        diffusion_steps=config.diffusion_steps,
    ).to(device)


MODEL_SPECS = {
    "graph-wavenet": ModelSpec(
        name="graph-wavenet",
        adapter_cls_name="GraphWaveNetAdapter",
        config_cls=GraphWaveNetConfig,
        build_model=build_graph_wavenet,
    ),
    "stgcn": ModelSpec(
        name="stgcn",
        adapter_cls_name="STGCNAdapter",
        config_cls=STGCNConfig,
        build_model=build_stgcn,
        build_scheduler=build_stgcn_scheduler,
    ),
    "dcrnn": ModelSpec(
        name="dcrnn",
        adapter_cls_name="DCRNNAdapter",
        config_cls=DCRNNConfig,
        build_model=build_dcrnn,
    ),
    "agcrn": ModelSpec(
        name="agcrn",
        adapter_cls_name="AGCRNAdapter",
        config_cls=AGCRNConfig,
        build_model=build_agcrn,
    ),
    "astgcn": ModelSpec(
        name="astgcn",
        adapter_cls_name="ASTGCNAdapter",
        config_cls=ASTGCNConfig,
        build_model=build_astgcn,
    ),
    "gman": ModelSpec(
        name="gman",
        adapter_cls_name="GMANAdapter",
        config_cls=GMANConfig,
        build_model=build_gman,
    ),
    "mtgnn": ModelSpec(
        name="mtgnn",
        adapter_cls_name="MTGNNAdapter",
        config_cls=MTGNNConfig,
        build_model=build_mtgnn,
    ),
    "stid": ModelSpec(
        name="stid",
        adapter_cls_name="STIDAdapter",
        config_cls=STIDConfig,
        build_model=build_stid,
    ),
    "dgcrn": ModelSpec(
        name="dgcrn",
        adapter_cls_name="DGCRNAdapter",
        config_cls=DGCRNConfig,
        build_model=build_dgcrn,
    ),
}

MODEL_REGISTRY = {}
for spec in MODEL_SPECS.values():
    MODEL_REGISTRY[spec.name] = type(
        spec.adapter_cls_name,
        (RegisteredForecastAdapter,),
        {
            "spec": spec,
            "model_name": spec.name,
            "config_cls": spec.config_cls,
        },
    )

CONFIG_REGISTRY = {name: spec.config_cls for name, spec in MODEL_SPECS.items()}

GraphWaveNetAdapter = MODEL_REGISTRY["graph-wavenet"]
STGCNAdapter = MODEL_REGISTRY["stgcn"]
DCRNNAdapter = MODEL_REGISTRY["dcrnn"]
AGCRNAdapter = MODEL_REGISTRY["agcrn"]
ASTGCNAdapter = MODEL_REGISTRY["astgcn"]
GMANAdapter = MODEL_REGISTRY["gman"]
MTGNNAdapter = MODEL_REGISTRY["mtgnn"]
STIDAdapter = MODEL_REGISTRY["stid"]
DGCRNAdapter = MODEL_REGISTRY["dgcrn"]

__all__ = [
    "AGCRNAdapter",
    "AGCRNConfig",
    "ASTGCNAdapter",
    "ASTGCNConfig",
    "CONFIG_REGISTRY",
    "DCRNNAdapter",
    "DCRNNConfig",
    "DGCRNAdapter",
    "DGCRNConfig",
    "GMANAdapter",
    "GMANConfig",
    "GraphWaveNetAdapter",
    "GraphWaveNetConfig",
    "MODEL_REGISTRY",
    "MODEL_SPECS",
    "MTGNNAdapter",
    "MTGNNConfig",
    "ModelSpec",
    "RegisteredForecastAdapter",
    "STGCNAdapter",
    "STGCNConfig",
    "STIDAdapter",
    "STIDConfig",
]
