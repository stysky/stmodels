# SpatialTemporal

> A unified spatiotemporal forecasting framework for traffic prediction, model reproduction, and fair cross-model comparison.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](./requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%2Bcu118-ee4c2c.svg)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Models-9-success.svg)](#当前支持)
[![Datasets](https://img.shields.io/badge/Datasets-2-success.svg)](#当前支持)
[![Stars](https://img.shields.io/github/stars/stysky/STmodels?style=social)](https://github.com/stysky/STmodels)

SpatialTemporal 是一个面向交通时空预测的统一实验框架。
这个仓库的重点不是只实现某一个模型，而是把数据读取、图结构处理、训练流程、实验配置和结果保存统一起来，让不同模型尽量在同一套实验口径下运行，方便复现、对比和继续扩展。

## 项目现状

- 已接入 9 个常见时空预测模型
- 已整理 2 个常用交通预测数据集：`METR-LA` 和 `PEMS-BAY`
- 支持统一 CLI 入口和统一 Python API
- 支持统一训练、导出原生数据产物和基于 checkpoint 的预测
- 训练结果会自动保存到 `runs/{model}/{dataset}/{timestamp-tag}/`

这个仓库目前更适合：

- 做统一口径下的模型复现和横向对比
- 快速建立交通时空预测 baseline
- 在现有训练框架上继续补充模型、配置或数据集

## 整体架构

从使用方式上看，这个项目可以分成 5 层：

1. 入口层：`scripts/run_experiment.py`
   负责解析命令行参数、读取配置、创建实验目录，并调用统一训练流程。
2. 编排层：`spatiotemporal/api.py`
   对外提供 `ExperimentRunner`，把数据集创建、模型创建、训练、导出和预测串起来。
3. 数据与配置层：`spatiotemporal/datasets.py`、`spatiotemporal/config_loader.py`
   负责数据集注册、原始数据读取、图结构解析，以及配置继承合并。
4. 统一训练层：`spatiotemporal/adapters/`
   负责把不同模型纳入同一套输入输出规范、训练循环、评估和导出流程。
5. 模型实现层：`spatiotemporal/models/`
   放置具体模型实现，例如 Graph WaveNet、STGCN、DCRNN、GMAN、DGCRN 等。

典型执行链路如下：

```text
configs/*.json or CLI args
  -> scripts/run_experiment.py
  -> ExperimentRunner
  -> dataset registry + config loader
  -> unified adapter pipeline
  -> model implementation
  -> metrics + run manager
  -> runs/... outputs
```

## 仓库结构

### 根目录

```text
SpatialTemporal/
  - README.md
  - requirements.txt
  - configs/
  - data/
  - scripts/
  - spatiotemporal/
```

关键文件和目录说明：

- `README.md`：项目首页说明，帮助第一次接触仓库的人快速了解项目定位和使用方式。
- `requirements.txt`：项目公共 Python 依赖，不包含环境相关的 PyTorch 安装命令。
- `configs/`：实验配置目录，每个 `json` 文件对应一个模型和数据集组合。
- `data/`：数据目录，包含原始时序文件和项目内部统一使用的图结构文件。
- `scripts/`：命令行脚本目录，当前核心入口是 `run_experiment.py`。
- `spatiotemporal/`：框架核心代码，包含数据结构、数据集、预处理、训练适配器、模型实现和高层 API。

### configs

```text
configs/
  - _base/common.json
  - graph-wavenet-metr-la.json
  - graph-wavenet-pems-bay.json
  - ...
```

- `configs/_base/common.json`：公共基础配置，当前主要保存数据集加载相关默认参数，例如图构建策略和缺失值处理方式。
- `configs/{model}-{dataset}.json`：具体实验入口配置，声明当前使用哪个模型、哪个数据集，以及必要的参数覆盖。

### data

```text
data/
  - README.md
  - raw/
  - graphs/
```

- `data/README.md`：补充说明数据目录约定。
- `data/raw/`：原始数据文件目录，例如 `metr-la.h5`、`pems-bay.h5` 以及官方图结构 `pkl` 文件。
- `data/graphs/`：项目内部统一使用的图结构文件目录，保存标准化后的 `npz` 图。

### scripts

```text
scripts/
  - run_experiment.py
```

- `scripts/run_experiment.py`：统一命令行入口。负责读取 `--config` 或 `--model/--dataset` 参数，构造实验目录，调用训练流程，并把结果写入 `runs/`。

### spatiotemporal

```text
spatiotemporal/
  - __init__.py
  - api.py
  - config_loader.py
  - core.py
  - datasets.py
  - metrics.py
  - preprocessing.py
  - run_manager.py
  - adapters/
  - models/
```

这里是项目最核心的部分：

- `spatiotemporal/__init__.py`：包导出入口，对外暴露 `ExperimentRunner`、`RunManager`、`TrafficData` 等常用对象。
- `spatiotemporal/api.py`：高层 API，定义 `ExperimentRunner`，是最推荐的 Python 调用入口。
- `spatiotemporal/config_loader.py`：负责加载 JSON 配置，并支持 `extends` 继承合并。
- `spatiotemporal/core.py`：定义核心数据结构，如 `TrafficData`、`SplitConfig`、`TrainingResult`。
- `spatiotemporal/datasets.py`：数据集注册中心和加载逻辑，当前内置 `METR-LA` 与 `PEMS-BAY`。
- `spatiotemporal/metrics.py`：评估指标实现，当前提供 `MAE`、`RMSE`、`MAPE` 等 masked 指标。
- `spatiotemporal/preprocessing.py`：预处理工具集合，负责读 H5、清洗缺失值、生成时间特征、切窗、图结构转换和保存等。
- `spatiotemporal/run_manager.py`：实验结果目录管理器，负责创建 `runs/...` 路径、保存配置文件和结果摘要。

### spatiotemporal/adapters

```text
spatiotemporal/adapters/
  - __init__.py
  - base.py
  - unified.py
  - model_zoo.py
```

这一层的作用是把“不同模型”统一到同一套实验流程里：

- `spatiotemporal/adapters/__init__.py`：模型注册和创建接口的导出入口，例如列出支持模型、根据名称创建 adapter。
- `spatiotemporal/adapters/base.py`：定义模型适配器抽象基类，以及设备解析、历史输入格式整理等基础能力。
- `spatiotemporal/adapters/unified.py`：统一训练主流程。负责构造特征、切分数据、标准化、创建 dataloader、训练、验证、导出和预测。
- `spatiotemporal/adapters/model_zoo.py`：模型注册表。定义每个模型对应的配置类、构造函数和调度器逻辑。

### spatiotemporal/models

```text
spatiotemporal/models/
  - __init__.py
  - common.py
  - conv_models.py
  - recurrent.py
  - attention_models.py
```

这一层是真正的模型实现：

- `spatiotemporal/models/__init__.py`：导出所有已注册模型类。
- `spatiotemporal/models/common.py`：多个模型共享的基础模块或通用组件。
- `spatiotemporal/models/conv_models.py`：卷积类和图卷积类模型实现，例如 `GraphWaveNet`、`STGCN`、`ASTGCN`、`MTGNN`。
- `spatiotemporal/models/recurrent.py`：递归和扩散类模型实现，例如 `DCRNN`、`AGCRN`、`DGCRN`。
- `spatiotemporal/models/attention_models.py`：注意力类模型实现，例如 `GMAN`、`STID`。

## 当前支持

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

如果官方图文件存在，框架会优先整理并使用 `data/graphs/*.npz` 中的标准化图结构文件。

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

## 建议阅读顺序

如果你是第一次看这个项目，推荐按下面顺序理解：

1. 先看 `README.md`，理解项目定位和目录结构。
2. 再看 `scripts/run_experiment.py`，了解实验是如何被启动的。
3. 接着看 `spatiotemporal/api.py`，理解 `ExperimentRunner` 如何调度数据集和模型。
4. 然后看 `spatiotemporal/adapters/unified.py`，理解统一训练流程。
5. 最后按需进入 `datasets.py`、`model_zoo.py` 和 `models/*.py` 看具体实现。

## 补充说明

- 当前仓库重点是统一实验流程，不包含可视化界面或部署模块。
- `data/raw/` 默认被 `.gitignore` 忽略，上传仓库时通常不会包含原始数据文件。
- 如果你的目标是快速跑通实验，从 `configs/` 和 `scripts/run_experiment.py` 开始就足够了。
