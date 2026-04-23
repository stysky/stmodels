# SpatialTemporal

> 一个面向交通时空预测的统一实验框架，用于模型复现、横向对比和统一训练管理。

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](./requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%2Bcu118-ee4c2c.svg)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Models-9-success.svg)](#当前支持)
[![Datasets](https://img.shields.io/badge/Datasets-6-success.svg)](#当前支持)
[![Stars](https://img.shields.io/github/stars/stysky/STmodels?style=social)](https://github.com/stysky/STmodels)

这个项目的重点不是单独实现某一个模型，而是把数据读取、图结构处理、训练流程、实验配置和结果保存统一起来，让不同模型尽量在同一套实验口径下运行。

## 项目结构

```text
SpatialTemporal/
|-- README.md                                  # 项目首页说明
|-- requirements.txt                           # Python 依赖列表
|
|-- configs/                                   # 实验配置目录
|   |-- _base/
|   |   `-- common.json                        # 公共基础配置
|   |-- {model}-{dataset}.json                 # 单个实验配置文件
|   `-- ...                                    # 其余模型/数据集组合配置
|
|-- data/                                      # 数据目录
|   |-- README.md                              # 数据目录约定说明
|   |-- raw/                                   # 固定的数据集目录约定
|   |   |-- dataset.template.json              # 自定义数据集模板
|   |   |-- METR-LA/
|   |   |   |-- metr-la.h5
|   |   |   `-- adj_METR-LA.pkl
|   |   |-- PEMS-BAY/
|   |   |   |-- pems-bay.h5
|   |   |   `-- adj_mx_bay.pkl
|   |   |-- PEMS03/
|   |   |   |-- PEMS03.npz
|   |   |   |-- PEMS03.csv
|   |   |   `-- PEMS03.txt
|   |   `-- MyDataset/
|   |       |-- dataset.json
|   |       |-- data.npz
|   |       |-- sensor_ids.txt
|   |       `-- edges.csv
|   `-- graphs/                                # 项目内部统一使用的图结构文件
|       |-- metr-la_adj.npz
|       |-- pems-bay_adj.npz
|       `-- {dataset}_adj.npz
|
|-- scripts/
|   `-- run_experiment.py                      # 统一命令行入口
|
|-- spatiotemporal/                            # 框架核心代码
|   |-- __init__.py                            # 导出常用接口
|   |-- api.py                                 # ExperimentRunner 高层 API
|   |-- config_loader.py                       # JSON 配置加载与 extends 合并
|   |-- core.py                                # TrafficData / TrainingResult 等核心数据结构
|   |-- datasets.py                            # 数据集注册、原始数据读取、图结构解析
|   |-- metrics.py                             # MAE / RMSE / MAPE 等评估指标
|   |-- preprocessing.py                       # 清洗、补值、时间特征、切窗、图工具
|   |-- run_manager.py                         # runs 目录创建与结果保存
|   |
|   |-- adapters/                              # 统一训练适配层
|   |   |-- __init__.py                        # 模型注册与创建入口
|   |   |-- base.py                            # 适配器抽象和基础工具
|   |   |-- unified.py                         # 统一 train / export / predict 主流程
|   |   `-- model_zoo.py                       # 模型配置类、构建函数、注册表
|   |
|   `-- models/                                # 具体模型实现
|       |-- __init__.py                        # 导出所有模型类
|       |-- common.py                          # 多模型共享模块
|       |-- conv_models.py                     # GraphWaveNet / STGCN / ASTGCN / MTGNN
|       |-- recurrent.py                       # DCRNN / AGCRN / DGCRN
|       `-- attention_models.py                # GMAN / STID
|
`-- runs/                                      # 训练输出目录，运行后自动生成
```

## 一句话看懂执行链路

```text
configs/*.json 或 CLI 参数
  -> scripts/run_experiment.py
  -> spatiotemporal.api.ExperimentRunner
  -> datasets + config_loader
  -> adapters.unified
  -> models/*
  -> metrics + run_manager
  -> runs/{model}/{dataset}/{timestamp-tag}/
```

## 当前支持

### 数据集

- `metr-la`
- `pems-bay`
- `pems03`
- `pems04`
- `pems07`
- `pems08`

也支持自动发现 `data/raw/<你的数据集>/dataset.json` 描述的自定义数据集。

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

`PyTorch` 需要根据你的 CUDA 或 CPU 环境单独安装。

### 2. 准备数据

项目现在固定使用 `data/raw/<DATASET>/` 目录结构：

```text
data/raw/
|-- dataset.template.json
|-- METR-LA/
|   |-- metr-la.h5
|   `-- adj_METR-LA.pkl
|-- PEMS-BAY/
|   |-- pems-bay.h5
|   `-- adj_mx_bay.pkl
|-- PEMS03/
|   |-- PEMS03.npz
|   |-- PEMS03.csv
|   `-- PEMS03.txt
|-- PEMS04/
|   |-- PEMS04.npz
|   `-- PEMS04.csv
|-- PEMS07/
|   |-- PEMS07.npz
|   `-- PEMS07.csv
`-- PEMS08/
    |-- PEMS08.npz
    `-- PEMS08.csv
```

框架支持的常见原始格式包括：

- 时序数据：`h5`、`npz`
- 传感器列表：`txt`、`csv`
- 图结构：`pkl`、稀疏 `npz`、边表 `csv`

如果官方图文件存在，框架会优先整理并使用 `data/graphs/*.npz` 中的标准化图结构文件。
更细的字段约定见 [data/README.md](/e:/Project%20Py/Pinjian_spatialTemporal/data/README.md:1)。

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
python scripts/run_experiment.py --model dcrnn --dataset pems03
python scripts/run_experiment.py --model stgcn --dataset pems08
```

如果你放入了自己的数据集目录，也可以直接运行：

```bash
python scripts/run_experiment.py --model dcrnn --dataset my-dataset
```

前提是 `data/raw/MyDataset/dataset.json` 里的 `name` 填的是 `my-dataset`。

### 4. 查看结果

训练完成后，结果会保存在：

```text
runs/{model}/{dataset}/{timestamp-tag}/
|-- checkpoints/best.pt
|-- exports/
|-- resolved_config.json
|-- result.json
`-- summary.txt
```

## 建议怎么读这个仓库

如果你是第一次看这个项目，按这个顺序最省时间：

1. 先看上面的“项目结构”，快速认识仓库分层。
2. 再看 `scripts/run_experiment.py`，理解实验从哪里启动。
3. 接着看 `spatiotemporal/api.py` 和 `spatiotemporal/adapters/unified.py`，理解统一训练流程。
4. 最后按需看 `datasets.py`、`model_zoo.py` 和 `models/` 里的具体实现。

## 补充说明

- 当前仓库重点是统一实验流程，不包含可视化界面或部署模块。
- `data/raw/` 下的真实原始数据默认被 `.gitignore` 忽略；仓库只保留模板文件 `data/raw/dataset.template.json`。
- 如果你的目标是快速跑通实验，从 `configs/` 和 `scripts/run_experiment.py` 开始就够了。
