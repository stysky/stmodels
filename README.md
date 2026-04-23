# SpatialTemporal

> A unified spatiotemporal forecasting framework for traffic prediction, model reproduction, and fair cross-model comparison.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](./requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%2Bcu118-ee4c2c.svg)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Models-9-success.svg)](#support-matrix)
[![Datasets](https://img.shields.io/badge/Datasets-2-success.svg)](#support-matrix)
[![Stars](https://img.shields.io/github/stars/stysky/STmodels?style=social)](https://github.com/stysky/STmodels)

SpatialTemporal 是一个面向交通时空预测的统一实验框架。
当前仓库已经打通了数据读取、图结构加载、统一训练入口、结果落盘和基础预测流程，目标是让不同模型尽量在同一套实验规范下运行，方便复现、对比和继续迭代。

## 项目现状

- 已接入 9 个常见时空预测模型，覆盖图卷积、递归、注意力和混合结构
- 已整理 2 个常用交通预测数据集：`METR-LA` 和 `PEMS-BAY`
- 提供统一命令行入口：`scripts/run_experiment.py`
- 提供统一 Python API：`spatiotemporal.ExperimentRunner`
- 训练输出会按 `runs/{model}/{dataset}/{timestamp-tag}/` 组织，便于管理实验结果
- 原始数据默认放在 `data/raw/`，项目内部统一使用的图结构文件放在 `data/graphs/`

这个仓库目前更适合以下工作：

- 做统一口径下的模型复现和横向对比
- 快速建立交通时空预测 baseline
- 在现有实验流程上继续补充模型、配置或数据集

## Support Matrix

### 数据集

- `metr-la`
- `pems-bay`

### 模型

- `graph-wavenet`
- `stgcn`
- `dcrnn`
- `agcrn`
- `astgcn`
- `gman`
- `mtgnn`
- `stid`
- `dgcrn`

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 只包含项目公共依赖，`PyTorch` 需要根据你的 CUDA 或 CPU 环境单独安装。

### 2. 准备数据

项目默认从 `data/raw/` 读取原始文件，当前约定的文件名如下：

```text
data/raw/
  - metr-la.h5
  - pems-bay.h5
  - adj_METR-LA.pkl
  - adj_mx_bay.pkl
```

如果图文件存在，框架会优先整理并使用 `data/graphs/*.npz` 中的标准化图结构文件。

### 3. 启动实验

使用配置文件运行：

```bash
python scripts/run_experiment.py --config configs/graph-wavenet-metr-la.json
python scripts/run_experiment.py --config configs/stgcn-pems-bay.json
```

直接指定模型和数据集运行：

```bash
python scripts/run_experiment.py --model dcrnn --dataset metr-la
python scripts/run_experiment.py --model mtgnn --dataset pems-bay
```

### 4. 查看结果

训练完成后，结果会保存在类似下面的目录中：

```text
runs/{model}/{dataset}/{timestamp-tag}/
  - checkpoints/best.pt
  - exports/
  - resolved_config.json
  - result.json
  - summary.txt
```

## 仓库结构

```text
SpatialTemporal/
  - configs/         # 实验配置
  - data/            # 数据与图结构文件
  - scripts/         # 命令行入口
  - spatiotemporal/  # 框架核心实现与统一 API
```

## 补充说明

- 当前仓库重点是统一实验流程，不包含可视化界面或部署模块
- `data/raw/` 默认被 `.gitignore` 忽略，上传仓库时通常不会包含原始数据文件
- 如果你只是第一次接触这个项目，从 `configs/`、`scripts/run_experiment.py` 和上面的运行示例开始就够了
