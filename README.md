# 时空交通预测统一框架

基于统一数据层和模型适配层的时空图神经网络实验框架。支持 `METR-LA`、`PEMS-BAY` 等交通数据集，支持 `Graph WaveNet`、`STGCN` 等模型，并提供统一的训练、导出、推理接口。

## 项目结构

```text
Pinjian_spatialTemporal/
│
├── README.md                   # 项目说明文档
├── requirements.txt            # 基础依赖
├── .gitignore                  # Git 忽略规则
│
├── configs/                    # 实验配置
│   ├── graph-wavenet-metr-la.json
│   ├── graph-wavenet-pems-bay.json
│   ├── stgcn-metr-la.json
│   └── stgcn-pems-bay.json
│
├── scripts/                    # 命令行入口
│   └── run_experiment.py       # 主入口：训练 / 导出原生数据 / 结果管理
│
├── data/                       # 数据资产
│   ├── README.md               # 数据目录说明
│   ├── raw/                    # 原始时序数据和原始图结构
│   └── graphs/                 # 项目内部统一使用的标准图结构文件
│
├── models/                     # 模型骨架源码
│   ├── README.md               # 模型目录说明
│   ├── Graph-WaveNet-master/
│   └── stgcn-main/
│
├── runs/                       # 训练产物（自动生成）
│   └── {model}/{dataset}/{timestamp-tag}/
│       ├── checkpoints/
│       │   └── best.pt
│       ├── exports/
│       ├── resolved_config.json
│       ├── result.json
│       └── summary.txt
│
└── spatiotemporal/             # 统一框架核心代码
    ├── __init__.py
    ├── api.py                  # 高层对外接口 ExperimentRunner
    ├── core.py                 # 核心数据结构 TrafficData / SplitConfig / TrainingResult
    ├── datasets.py             # 数据集注册与加载
    ├── preprocessing.py        # 通用预处理、图变换、滑窗
    ├── metrics.py              # 指标函数
    ├── run_manager.py          # 运行目录与结果保存
    └── adapters/               # 模型适配层
        ├── __init__.py
        ├── base.py
        ├── graph_wavenet.py
        └── stgcn.py
```

## 项目目标

- 统一接入不同时空交通数据集
- 统一封装不同模型的输入输出接口
- 统一图结构处理、切分、滑窗与训练流程
- 对外提供稳定的 Python 调用接口
- 保留原模型骨架，避免重复维护多套训练脚本

## 当前支持

### 数据集

- `metr-la`
- `pems-bay`

### 模型

- `graph-wavenet`
- `stgcn`

## 环境依赖

```bash
pip install -r requirements.txt
```

基础依赖主要包括：

```text
numpy
pandas
scipy
scikit-learn
h5py
```

`PyTorch` 需要根据你的 CPU / CUDA 环境单独安装。

## 快速开始

### 1. 命令行运行实验

训练 `Graph WaveNet + METR-LA`：

```bash
python scripts/run_experiment.py --config configs/graph-wavenet-metr-la.json
```

训练 `STGCN + PEMS-BAY`：

```bash
python scripts/run_experiment.py --config configs/stgcn-pems-bay.json
```

只导出 `Graph WaveNet` 原生数据：

```bash
python scripts/run_experiment.py --config configs/graph-wavenet-pems-bay.json --export_dir artifacts/pems_bay_gwn --export_only
```

### 2. Python 高层接口调用

项目对外推荐使用 `ExperimentRunner`：

```python
from pathlib import Path

from spatiotemporal import ExperimentRunner
from spatiotemporal.adapters import GraphWaveNetConfig

runner = ExperimentRunner(Path(r"e:\Project Py\Pinjian_spatialTemporal"))

result = runner.train(
    model_name="graph-wavenet",
    dataset_name="metr-la",
    config=GraphWaveNetConfig(epochs=10),
)
```

### 3. 导出模型原生数据

```python
export_result = runner.export(
    model_name="graph-wavenet",
    dataset_name="pems-bay",
    config=GraphWaveNetConfig(),
    output_dir=r"artifacts\pems_bay_gwn",
)
```

这里的“导出原生模型数据”不是导出模型权重，而是把统一框架中的数据导出成原模型仓库可直接使用的文件格式。

例如对 `Graph WaveNet`，通常导出：

- `train.npz`
- `val.npz`
- `test.npz`
- `adj.pkl`

### 4. 加载 checkpoint 推理

```python
prediction = runner.predict(
    model_name="graph-wavenet",
    dataset_name="metr-la",
    checkpoint_path=result.checkpoint_path,
    history=history_df,
    config=GraphWaveNetConfig(),
)
```

## 对外统一接口

推荐直接使用以下 3 个高层接口：

```python
runner.train(model_name, dataset_name, config)
runner.export(model_name, dataset_name, config, output_dir)
runner.predict(model_name, dataset_name, checkpoint_path, history, config)
```

对于大多数调用方，只需要关心：

- `model_name`
- `dataset_name`
- `config`

不需要自己处理：

- 原始图文件格式
- 数据切分
- 滑窗
- 特征扩展
- 输入 shape 转换
- checkpoint 加载

这些都由框架内部完成。

## 数据与图结构说明

### 原始时序数据

当前两个数据集的原始时序本质都是：

- 速度矩阵 `[T, N]`

其中：

- `T` 表示时间步数
- `N` 表示节点数

例如 `METR-LA` 的 `metr-la.h5` 读出来后，本质上是：

- 行：时间步
- 列：传感器节点
- 单元格：该时间步该节点的速度

### 图结构

项目内部统一使用：

- `data/graphs/*.npz`

当前约定：

- `METR-LA`
  - 官方图来源：`data/raw/adj_METR-LA.pkl`
  - 内部标准图：`data/graphs/metr-la_adj.npz`
- `PEMS-BAY`
  - 官方图来源：`data/raw/adj_mx_bay.pkl`
  - 内部标准图：`data/graphs/pems-bay_adj.npz`

模型内部不会直接使用原始图文件，而是：

- `Graph WaveNet`
  - `adjacency -> supports`
- `STGCN`
  - `adjacency -> gso`

### 官方图与标准图一致性校验

当项目接入新的官方图文件时，建议先做一次一致性检查，再决定是否替换当前标准图。

建议至少检查下面几项：

- 矩阵 shape 是否一致
- 节点数量是否一致
- 传感器顺序是否一致
- 非零元素数量是否一致
- 是否对称是否一致
- `max abs diff` 是否接近 `0`

如果结果显示两份图不一致，应以官方图为基准，并重新生成项目内部标准图。

`PEMS-BAY` 已经按这个流程处理过：

- 官方图：`data/raw/adj_mx_bay.pkl`
- 标准图：`data/graphs/pems-bay_adj.npz`
- 当前结果：两者核心矩阵完全一致

推荐校验流程：

```text
官方 pkl
  → load_pickle_adj()
  → 比较现有 data/graphs/*.npz
  → 若不一致，则 convert_adj_pkl_to_npz()
  → 更新项目内部标准图
```

## 数据流

```text
原始 h5 / pkl / npz
  → spatiotemporal/datasets.py        读取原始数据与图结构
  → TrafficData                       统一数据对象
  → adapters/*.py                    转成各模型所需输入格式
  → models/*                         模型前向传播
  → 训练 / 指标 / checkpoint
  → runs/                            保存结果
```

更具体地说：

```text
METR-LA / PEMS-BAY
  → read_traffic_h5()                读取时序数据
  → clean_speed_dataframe()          缺失值处理
  → _resolve_adjacency()             图结构解析
  → TrafficData(values=[T,N], adjacency=[N,N])
  → GraphWaveNetAdapter / STGCNAdapter
  → prepare_training_bundle()
  → train() / predict()
```

## 模型接入格式

### Graph WaveNet

```text
TrafficData.values [T, N]
  → 切分后 [T_split, N]
  → 特征扩展后 [T_split, N, F]
  → 滑窗后 x: [B, T_in, N, F], y: [B, T_out, N, F]
  → 进模型前转为 [B, F, N, T]
```

默认情况下：

- `F = 2`
- 特征包括：
  - `speed`
  - `time_in_day`

### STGCN

```text
TrafficData.values [T, N]
  → 切分后 [T_split, N]
  → 滑窗后 x: [B, 1, T_in, N], y: [B, N]
```

当前这版 `STGCN` 是：

- 输入过去 `history_steps`
- 预测未来第 `prediction_step` 个时间点

不是输出未来整段序列。

## 推理输入要求

### Graph WaveNet

- `history` 至少包含最近 `history_steps` 行
- 推荐使用 `pandas.DataFrame`
- 列顺序必须与数据集 `sensor_ids` 一致
- 输出 shape 为 `[horizon_steps, num_nodes]`

### STGCN

- `history` 至少包含最近 `history_steps` 行
- 推荐使用 `pandas.DataFrame`
- 列顺序必须与数据集 `sensor_ids` 一致
- 输出 shape 为 `[num_nodes]`

## 输出说明

### 训练输出 (`runs/`)

```text
runs/{model}/{dataset}/{timestamp-tag}/
├── checkpoints/
│   └── best.pt
├── exports/
├── resolved_config.json
├── result.json
└── summary.txt
```

其中：

- `best.pt`
  - 验证集最优 checkpoint
- `resolved_config.json`
  - 最终生效配置
- `result.json`
  - 训练结果与指标
- `summary.txt`
  - 简要结果摘要

### 导出原生模型数据

根据模型不同，导出文件格式不同。

例如：

- `Graph WaveNet`
  - `train.npz`
  - `val.npz`
  - `test.npz`
  - `adj.pkl`

- `STGCN`
  - `vel.csv`
  - `adj.npz`

## 关键核心代码

如果你要掌握项目主线，建议优先看这些文件：

- `spatiotemporal/api.py`
  - 对外高层接口
- `spatiotemporal/core.py`
  - 核心数据结构
- `spatiotemporal/datasets.py`
  - 数据集统一加载
- `spatiotemporal/preprocessing.py`
  - 预处理与图变换
- `spatiotemporal/adapters/graph_wavenet.py`
  - Graph WaveNet 适配逻辑
- `spatiotemporal/adapters/stgcn.py`
  - STGCN 适配逻辑
- `scripts/run_experiment.py`
  - 命令行入口

推荐学习顺序：

```text
README.md
  → api.py
  → run_experiment.py
  → datasets.py
  → preprocessing.py
  → adapters/*.py
  → models/*
```

## 扩展新数据集

1. 在 `spatiotemporal/datasets.py` 中增加新的数据集类
2. 注册到 `DATASET_REGISTRY`
3. 实现原始时序解析与图结构解析
4. 复用现有模型适配器，或按需要新增逻辑

## 扩展新模型

1. 在 `spatiotemporal/adapters/` 中增加新的 adapter
2. 实现以下接口：
   - `prepare_training_bundle`
   - `export_native_artifacts`
   - `train`
   - `load_checkpoint`
   - `predict`
3. 注册到 `spatiotemporal/adapters/__init__.py`
