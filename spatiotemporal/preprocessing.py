from __future__ import annotations

import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import norm

from .core import SplitConfig


class StandardScaler:
    def __init__(self, mean: np.ndarray | float, std: np.ndarray | float, eps: float = 1e-6):
        self.mean = mean
        self.std = np.maximum(std, eps)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def clean_speed_dataframe(
    df: pd.DataFrame,
    treat_zero_as_missing: bool = True,
    fill_method: str = "interpolate",
) -> pd.DataFrame:
    if not treat_zero_as_missing:
        return df

    cleaned = df.mask(df == 0)
    if fill_method == "interpolate":
        cleaned = cleaned.interpolate(method="linear", axis=0, limit_direction="both")
        cleaned = cleaned.ffill().bfill()
    elif fill_method == "ffill":
        cleaned = cleaned.ffill().bfill()
    elif fill_method == "bfill":
        cleaned = cleaned.bfill().ffill()
    else:
        raise ValueError(f"Unsupported fill_method: {fill_method}")

    remaining_missing = int(cleaned.isna().sum().sum())
    if remaining_missing:
        raise ValueError(f"Still have {remaining_missing} missing values after filling")
    return cleaned


def read_traffic_h5(
    h5_path: str | Path,
    start_date: str,
    freq: str = "5min",
) -> tuple[pd.DataFrame, list[str]]:
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as handle:
        root = "df" if "df" in handle else "speed"
        values = handle[f"{root}/block0_values"][:]
        sensor_ids = handle[f"{root}/block0_items"][:]

    if values.ndim == 3:
        values = values[:, :, 0]
    elif values.ndim == 4:
        values = values[:, :, 0, 0]

    sensor_ids = [sid.decode() if isinstance(sid, bytes) else str(sid) for sid in sensor_ids]
    df = pd.DataFrame(values, columns=sensor_ids)
    df.index = pd.date_range(start=start_date, periods=len(df), freq=freq)
    return df, sensor_ids


def split_indices(num_items: int, split_config: SplitConfig) -> tuple[slice, slice, slice]:
    num_train = round(num_items * split_config.train_ratio)
    num_test = round(num_items * split_config.test_ratio)
    num_val = num_items - num_train - num_test

    train_slice = slice(0, num_train)
    val_slice = slice(num_train, num_train + num_val)
    test_slice = slice(num_train + num_val, num_items)
    return train_slice, val_slice, test_slice


def add_graph_wavenet_features(
    df: pd.DataFrame,
    add_time_in_day: bool = True,
    add_day_in_week: bool = False,
) -> np.ndarray:
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day.astype(np.float32))

    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled.astype(np.float32))

    return np.concatenate(feature_list, axis=-1).astype(np.float32)


def make_graph_wavenet_windows(
    df: pd.DataFrame,
    history_steps: int,
    horizon_steps: int,
    y_start: int = 1,
    add_time_in_day: bool = True,
    add_day_in_week: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = add_graph_wavenet_features(
        df,
        add_time_in_day=add_time_in_day,
        add_day_in_week=add_day_in_week,
    )
    x_offsets = np.sort(np.arange(-(history_steps - 1), 1, 1))
    y_offsets = np.sort(np.arange(y_start, horizon_steps + 1, 1))

    min_t = abs(min(x_offsets))
    max_t = abs(data.shape[0] - abs(max(y_offsets)))
    x, y = [], []
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    return (
        np.stack(x, axis=0).astype(np.float32),
        np.stack(y, axis=0).astype(np.float32),
        x_offsets.astype(np.int64),
        y_offsets.astype(np.int64),
    )


def make_stgcn_windows(data: np.ndarray, history_steps: int, prediction_step: int) -> tuple[np.ndarray, np.ndarray]:
    n_vertex = data.shape[1]
    total = len(data) - history_steps - prediction_step
    if total <= 0:
        raise ValueError("Not enough samples to create STGCN windows")

    x = np.zeros((total, 1, history_steps, n_vertex), dtype=np.float32)
    y = np.zeros((total, n_vertex), dtype=np.float32)
    for i in range(total):
        head = i
        tail = i + history_steps
        x[i, :, :, :] = data[head:tail].reshape(1, history_steps, n_vertex)
        y[i] = data[tail + prediction_step - 1]
    return x, y


def load_pickle_adj(pickle_file: str | Path) -> tuple[list[str], dict[str, int], np.ndarray]:
    with open(pickle_file, "rb") as handle:
        try:
            sensor_ids, sensor_id_to_ind, adj = pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            sensor_ids, sensor_id_to_ind, adj = pickle.load(handle, encoding="latin1")
    return sensor_ids, sensor_id_to_ind, np.asarray(adj, dtype=np.float32)


def load_adjacency_npz(npz_file: str | Path) -> sp.csc_matrix:
    return sp.load_npz(npz_file).tocsc()


def save_adj_pkl(output_path: str | Path, sensor_ids: list[str], adj_mx: np.ndarray) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sensor_id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}
    with open(output_path, "wb") as handle:
        pickle.dump((sensor_ids, sensor_id_to_ind, adj_mx), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def save_adj_npz(output_path: str | Path, adj_mx: sp.spmatrix | np.ndarray) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if sp.issparse(adj_mx):
        sparse_adj = adj_mx.tocsc()
    else:
        sparse_adj = sp.csc_matrix(np.asarray(adj_mx, dtype=np.float32))
    sp.save_npz(output_path, sparse_adj)
    return output_path


def convert_adj_pkl_to_npz(pkl_path: str | Path, npz_path: str | Path) -> tuple[Path, list[str]]:
    sensor_ids, _, adj_mx = load_pickle_adj(pkl_path)
    save_adj_npz(npz_path, adj_mx)
    return Path(npz_path), sensor_ids


def build_correlation_adjacency(
    values: np.ndarray,
    top_k: int = 10,
    min_weight: float = 0.1,
    symmetric: bool = True,
    self_loop_weight: float | None = 1.0,
) -> np.ndarray:
    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 0.0)

    if min_weight > 0:
        corr[corr < min_weight] = 0.0
    if 0 < top_k < corr.shape[1]:
        pruned = np.zeros_like(corr)
        for i in range(corr.shape[0]):
            row = corr[i]
            idx = np.argpartition(row, -top_k)[-top_k:]
            pruned[i, idx] = row[idx]
        corr = pruned

    if symmetric:
        corr = np.maximum(corr, corr.T)
    if self_loop_weight is not None:
        np.fill_diagonal(corr, self_loop_weight)
    return corr.astype(np.float32)


def asym_adj(adj: np.ndarray) -> np.ndarray:
    sparse_adj = sp.coo_matrix(adj)
    rowsum = np.array(sparse_adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return d_mat.dot(sparse_adj).astype(np.float32).todense()


def sym_adj(adj: np.ndarray) -> np.ndarray:
    sparse_adj = sp.coo_matrix(adj)
    rowsum = np.array(sparse_adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sparse_adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def calculate_normalized_laplacian(adj: np.ndarray) -> sp.coo_matrix:
    sparse_adj = sp.coo_matrix(adj)
    d = np.array(sparse_adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(sparse_adj.shape[0]) - sparse_adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def calculate_scaled_laplacian(adj_mx: np.ndarray, lambda_max: float = 2.0, undirected: bool = True) -> np.ndarray:
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    laplacian = calculate_normalized_laplacian(adj_mx)
    laplacian = sp.csr_matrix(laplacian)
    identity = sp.identity(laplacian.shape[0], format="csr", dtype=laplacian.dtype)
    scaled = (2 / lambda_max * laplacian) - identity
    return scaled.astype(np.float32).todense()


def graph_wavenet_supports(adj_mx: np.ndarray, adjtype: str) -> list[np.ndarray]:
    if adjtype == "scalap":
        return [calculate_scaled_laplacian(adj_mx)]
    if adjtype == "normlap":
        return [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    if adjtype == "symnadj":
        return [sym_adj(adj_mx)]
    if adjtype == "transition":
        return [asym_adj(adj_mx)]
    if adjtype == "doubletransition":
        return [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    if adjtype == "identity":
        return [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    raise ValueError(f"Unsupported Graph WaveNet adjacency type: {adjtype}")


def calc_gso(dir_adj: sp.spmatrix | np.ndarray, gso_type: str) -> sp.spmatrix:
    if not sp.issparse(dir_adj):
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != "csc":
        dir_adj = dir_adj.tocsc()

    n_vertex = dir_adj.shape[0]
    identity = sp.identity(n_vertex, format="csc")
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)

    if gso_type in {"sym_renorm_adj", "rw_renorm_adj", "sym_renorm_lap", "rw_renorm_lap"}:
        adj = adj + identity

    if gso_type in {"sym_norm_adj", "sym_renorm_adj", "sym_norm_lap", "sym_renorm_lap"}:
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.0
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format="csc")
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)
        if gso_type in {"sym_norm_lap", "sym_renorm_lap"}:
            return identity - sym_norm_adj
        return sym_norm_adj

    if gso_type in {"rw_norm_adj", "rw_renorm_adj", "rw_norm_lap", "rw_renorm_lap"}:
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.0
        deg_inv = np.diag(row_sum_inv)
        rw_norm_adj = deg_inv.dot(adj)
        if gso_type in {"rw_norm_lap", "rw_renorm_lap"}:
            return identity - rw_norm_adj
        return rw_norm_adj

    raise ValueError(f"{gso_type} is not defined")


def calc_chebynet_gso(gso: sp.spmatrix | np.ndarray) -> sp.spmatrix:
    if not sp.issparse(gso):
        gso = sp.csc_matrix(gso)
    elif gso.format != "csc":
        gso = gso.tocsc()

    identity = sp.identity(gso.shape[0], format="csc")
    eigval_max = norm(gso, 2)
    if eigval_max >= 2:
        return gso - identity
    return 2 * gso / eigval_max - identity
