from .graph_wavenet import GraphWaveNetAdapter, GraphWaveNetConfig
from .stgcn import STGCNAdapter, STGCNConfig

MODEL_REGISTRY = {
    "graph-wavenet": GraphWaveNetAdapter,
    "stgcn": STGCNAdapter,
}

__all__ = [
    "GraphWaveNetAdapter",
    "GraphWaveNetConfig",
    "MODEL_REGISTRY",
    "STGCNAdapter",
    "STGCNConfig",
]
