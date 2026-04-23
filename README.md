# SpatialTemporal

> A unified spatiotemporal forecasting framework for traffic prediction, model reproduction, and fair cross-model comparison.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](./requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%2Bcu118-ee4c2c.svg)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Models-9-success.svg)](#supported-models)
[![Datasets](https://img.shields.io/badge/Datasets-2-success.svg)](#supported-datasets)
[![Stars](https://img.shields.io/github/stars/stysky/STmodels?style=social)](https://github.com/stysky/STmodels)

SpatialTemporal 是一个面向交通时空预测的统一实验框架。  
它的核心目标不是只实现某一个模型，而是把不同数据集、不同模型和不同训练脚本收敛到同一套标准下，让复现、对比和扩展都更简单。

## Overview

- 用统一的数据表示、训练流程和配置体系管理多种时空预测模型
- 适合做模型复现、横向 benchmark、消融实验和新模型接入
- 已经内置 `Graph WaveNet`、`STGCN`、`DCRNN`、`AGCRN`、`MTGNN` 等常见模型
- `Graph WaveNet` 和 `STGCN` 已整理为项目内置实现，不再依赖外部 vendor 目录运行

## Why This Repo

很多时空预测项目都有同样的问题：

- 每个模型一个仓库，训练入口和数据处理方式都不一样
- 很难在同一套设定下公平比较不同模型
- 加一个新模型，往往要复制一整套脚本和目录结构

这个项目想解决的就是这些重复和分裂：

- `统一输入输出规范`：所有模型都对齐到相同的张量接口
- `统一训练管线`：`train / export / predict` 走同一套逻辑
- `统一配置体系`：默认参数在代码里，实验配置只保留必要覆盖
- `统一扩展方式`：新增模型主要补 `Config + build_model + 注册`

## Table of Contents

- [Highlights](#highlights)
- [Supported Datasets](#supported-datasets)
- [Supported Models](#supported-models)
- [Quick Start](#quick-start)
- [Python API](#python-api)
- [Project Structure](#project-structure)
- [Data Convention](#data-convention)
- [Configuration Strategy](#configuration-strategy)
- [Training Outputs](#training-outputs)
- [How to Extend](#how-to-extend)
- [Validation Status](#validation-status)

## Highlights

- Unified data representation: `TrafficData.values` uses `[T, N, F]`
- Unified supervised learning window: input `x = [B, T, N, F]`, output `y = [B, H, N, 1]`
- Unified training entry: script, API, and adapters share the same pipeline
- Unified model registry: all models are created through `ModelSpec -> Config -> Adapter`
- Unified config strategy: model defaults stay in code, experiment JSON stays lightweight
- Unified extension path: adding a dataset or model does not require cloning a new training script

## Supported Datasets

- `metr-la`
- `pems-bay`

## Supported Models

- `graph-wavenet`
- `stgcn`
- `dcrnn`
- `agcrn`
- `astgcn`
- `gman`
- `mtgnn`
- `stid`
- `dgcrn`

所有模型都对齐到了同样的输入输出规范，便于在同一套实验设置下做横向比较。

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

然后根据你的环境单独安装带 CUDA 的 PyTorch。

### 2. Train with a config file

```bash
python scripts/run_experiment.py --config configs/graph-wavenet-metr-la.json
python scripts/run_experiment.py --config configs/stgcn-metr-la.json
python scripts/run_experiment.py --config configs/agcrn-metr-la.json
```

### 3. Train by directly specifying model and dataset

```bash
python scripts/run_experiment.py --model dcrnn --dataset metr-la
python scripts/run_experiment.py --model mtgnn --dataset pems-bay
```

### 4. Export unified artifacts only

```bash
python scripts/run_experiment.py --config configs/gman-metr-la.json --export_only
```

默认会导出：

- `train.npz`
- `val.npz`
- `test.npz`
- `adjacency.npz`
- `metadata.json`

## Python API

推荐通过 `ExperimentRunner` 调用高层接口。

### Train

```python
from pathlib import Path

from spatiotemporal import ExperimentRunner

runner = ExperimentRunner(Path(r"/path/to/SpatialTemporal"))

result = runner.train(
    model_name="graph-wavenet",
    dataset_name="metr-la",
)

print(result.metrics)
```

### Export

```python
export_result = runner.export(
    model_name="agcrn",
    dataset_name="metr-la",
)
```

### Predict

```python
prediction = runner.predict(
    model_name="graph-wavenet",
    dataset_name="metr-la",
    checkpoint_path=result.checkpoint_path,
    history=history_df,
)
```

推理输出 shape：

- `[H, N, 1]`

## Project Structure

```text
SpatialTemporal/
├── configs/                    # Experiment configs
├── data/                       # Data assets
├── scripts/
│   └── run_experiment.py       # Unified CLI entry
└── spatiotemporal/
    ├── adapters/
    │   ├── base.py             # Adapter abstraction and shared input normalization
    │   ├── unified.py          # Shared train / export / predict workflow
    │   ├── model_zoo.py        # Model configs, builders, and specs
    │   └── __init__.py         # Registry and creation helpers
    ├── models/                 # Model implementations
    ├── api.py                  # High-level API: ExperimentRunner
    ├── config_loader.py        # JSON config loader with extends support
    ├── core.py                 # Core data structures
    ├── datasets.py             # Dataset registry and loading
    ├── metrics.py              # Evaluation metrics
    ├── preprocessing.py        # Preprocessing, windowing, graph utilities
    └── run_manager.py          # Run directory and result management
```

## Data Convention

### Raw data

框架内部统一使用 `TrafficData`：

- `values`: `[T, N, F]`
- `T`: 时间步数
- `N`: 节点数
- `F`: 特征维度

当前交通数据的基础目标特征为：

- `speed`

训练前会自动补充时间特征，例如：

- `time_of_day_sin`
- `time_of_day_cos`
- `day_of_week_sin`
- `day_of_week_cos`

因此模型输入通常会变成：

- `x: [B, history_steps, N, F]`

### Supervised learning window

所有模型统一使用：

- Input: `x = [B, T, N, F]`
- Output: `y = [B, H, N, 1]`

这让不同模型之间可以复用同样的数据管线和评估逻辑。

## Configuration Strategy

当前配置不再是“每个模型 + 每个数据集都复制一整份参数”。

现在的原则是：

- 模型默认训练参数定义在各自的 `Config` dataclass 中
- `configs/*.json` 主要声明 `model`、`dataset`
- 公共配置放到共享基配置里，通过 `extends` 继承
- 只有需要覆盖默认值时，才额外写 `model_config`

示例：

```json
{
  "extends": "_base/common.json",
  "model": "stgcn",
  "dataset": "metr-la"
}
```

如果需要覆盖训练轮数或模型参数：

```json
{
  "extends": "_base/common.json",
  "model": "stgcn",
  "dataset": "metr-la",
  "model_config": {
    "epochs": 5,
    "droprate": 0.3
  }
}
```

## Training Outputs

默认输出目录：

```text
runs/{model}/{dataset}/{timestamp-tag}/
├── checkpoints/best.pt
├── exports/
├── resolved_config.json
├── result.json
└── summary.txt
```

含义如下：

- `best.pt`: 验证集最优 checkpoint
- `resolved_config.json`: 本次运行最终生效的配置
- `result.json`: 训练结果和指标
- `summary.txt`: 简要摘要

## How to Extend

### Add a new dataset

1. 在 `spatiotemporal/datasets.py` 中新增数据集类
2. 实现原始时序读取逻辑
3. 实现图结构解析逻辑
4. 注册到 `DATASET_REGISTRY`

只要能输出统一的 `TrafficData`，后续训练流程通常不需要额外改动。

### Add a new model

1. 在 `spatiotemporal/models/` 中实现模型前向
2. 保持统一接口：

```python
forward(x: [B, T, N, F]) -> y: [B, H, N, 1]
```

3. 在 `spatiotemporal/adapters/model_zoo.py` 中补充：

- `Config`
- `build_model`
- 必要时补 `build_scheduler`
- 注册到 `MODEL_SPECS`

## Validation Status

当前版本已完成基础烟雾验证，覆盖：

- 9 个模型的统一创建与注册检查
- 9 个模型在 `METR-LA` 上的前向 shape 检查
- 配置继承加载检查
- CLI `--help` 检查
- `STGCN` 和 `Graph WaveNet` 的最小训练主链路检查

在本地 `E:\conda_envs\pt312` 环境下，已确认：

- `torch==2.7.1+cu118`
- `torch.cuda.is_available() == True`

## Use Cases

当前项目主要适用于：

- 交通流 / 速度预测
- 时空图预测任务
- 多模型统一对比实验
- 后续接入更多时空模型与数据集

如果这个项目对你有帮助，欢迎点个 Star，也欢迎提 Issue 或 PR。
