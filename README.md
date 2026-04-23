# Pinjian SpatialTemporal

统一的时空预测实验框架，目标是把不同数据集、不同模型和统一训练流程收敛到同一套接口下，方便做复现实验、横向对比和后续扩展。

结构调整记录见 [MIGRATION.md](/e:/Project%20Py/Pinjian_spatialTemporal/MIGRATION.md:1)。

## 当前特点

- 统一数据表示：`TrafficData.values` 使用 `[T, N, F]`
- 统一监督学习窗口：输入 `x=[B, T, N, F]`，输出 `y=[B, H, N, 1]`
- 统一训练入口：脚本、API、adapter 共用同一套 bundle / train / export / predict 流程
- 统一模型注册：所有模型都通过同一套 `ModelSpec -> Config -> Adapter` 注册
- 统一配置策略：模型默认参数放在对应 `Config` dataclass，实验 JSON 只保留必要覆盖项
- 统一扩展方式：新增模型或数据集时，不需要再复制一整套训练脚本

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

其中：

- `Graph WaveNet` 和 `STGCN` 已整理为项目内置实现
- 其余模型使用框架内统一实现
- 所有模型都对齐到了相同的输入输出规范

## 项目结构

```text
Pinjian_spatialTemporal/
├── configs/                    # 实验配置
├── data/                       # 数据资源
├── runs/                       # 训练输出
├── scripts/
│   └── run_experiment.py       # 统一命令行入口
└── spatiotemporal/
    ├── adapters/
    │   ├── base.py             # adapter 抽象与公共输入规整
    │   ├── unified.py          # 通用训练/导出/预测流程
    │   ├── model_zoo.py        # 模型配置、构建函数、注册规格
    │   └── __init__.py         # 注册表与创建入口
    ├── models/                 # 模型实现
    ├── api.py                  # 高层接口 ExperimentRunner
    ├── config_loader.py        # JSON 配置加载与 extends 合并
    ├── core.py                 # 核心数据结构
    ├── datasets.py             # 数据集注册与加载
    ├── metrics.py              # 指标函数
    ├── preprocessing.py        # 预处理、切窗、图处理
    └── run_manager.py          # 运行目录与结果保存
```

## 统一数据约定

### 原始数据

框架内部统一使用 `TrafficData`：

- `values`: `[T, N, F]`
- `T`: 时间步数
- `N`: 节点数
- `F`: 特征维度

当前交通数据的基础目标特征是：

- `speed`

训练前会自动补充时间特征，例如：

- `time_of_day_sin`
- `time_of_day_cos`
- `day_of_week_sin`
- `day_of_week_cos`

因此模型训练输入通常变为：

- `x: [B, history_steps, N, F]`

### 监督学习窗口

所有模型统一使用：

- 输入：`x = [B, T, N, F]`
- 输出：`y = [B, H, N, 1]`

这意味着：

- `STGCN` 已不再使用旧的单步预测接口
- 所有模型都遵循同一套 shape 约定

## 配置策略

当前配置不再是“每个模型 + 每个数据集都复制一整份参数”。

现在的原则是：

- 模型默认训练参数定义在各自的 `Config` dataclass 中
- `configs/*.json` 主要声明 `model`、`dataset`
- 公共配置放到共享基配置里，通过 `extends` 继承
- 只有需要覆盖默认值时，才额外写 `model_config`

例如：

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

`extends` 支持相对路径，也支持多层继承，后加载项会覆盖前面的值。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

然后按本机环境单独安装带 CUDA 的 PyTorch。

### 2. 通过配置文件训练

```bash
python scripts/run_experiment.py --config configs/graph-wavenet-metr-la.json
python scripts/run_experiment.py --config configs/stgcn-metr-la.json
python scripts/run_experiment.py --config configs/agcrn-metr-la.json
```

### 3. 直接指定模型与数据集

```bash
python scripts/run_experiment.py --model dcrnn --dataset metr-la
python scripts/run_experiment.py --model mtgnn --dataset pems-bay
```

### 4. 只导出统一格式工件

```bash
python scripts/run_experiment.py --config configs/gman-metr-la.json --export_only
```

默认导出内容通常包括：

- `train.npz`
- `val.npz`
- `test.npz`
- `adjacency.npz`
- `metadata.json`

## Python 接口

推荐使用高层接口 `ExperimentRunner`。

### 训练

```python
from pathlib import Path

from spatiotemporal import ExperimentRunner

runner = ExperimentRunner(Path(r"e:\Project Py\Pinjian_spatialTemporal"))

result = runner.train(
    model_name="graph-wavenet",
    dataset_name="metr-la",
)

print(result.metrics)
```

### 导出

```python
export_result = runner.export(
    model_name="agcrn",
    dataset_name="metr-la",
)
```

### 推理

```python
prediction = runner.predict(
    model_name="graph-wavenet",
    dataset_name="metr-la",
    checkpoint_path=result.checkpoint_path,
    history=history_df,
)
```

推理输出 shape 为：

- `[H, N, 1]`

## 数据流

```text
原始 h5 / pkl / npz
  -> datasets.py 读取原始时序与图结构
  -> TrafficData(values=[T,N,F])
  -> preprocessing.py 生成时间特征、切窗、标准化
  -> adapters/ 组织 training bundle
  -> models/ 前向计算
  -> train / export / predict
```

## 训练输出

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

- `best.pt`：验证集最优 checkpoint
- `resolved_config.json`：本次运行最终生效的配置
- `result.json`：训练结果和指标
- `summary.txt`：简要摘要

## 扩展方式

### 新增数据集

1. 在 `spatiotemporal/datasets.py` 中新增数据集类
2. 实现原始时序读取逻辑
3. 实现图结构解析逻辑
4. 注册到 `DATASET_REGISTRY`

只要能输出统一的 `TrafficData`，后续训练流程通常不需要额外改动。

### 新增模型

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

## 当前验证情况

当前版本已完成基础烟雾验证，覆盖：

- 9 个模型的统一创建与注册检查
- 9 个模型在 `METR-LA` 上的前向 shape 检查
- 配置继承加载检查
- CLI `--help` 检查

在本地 `E:\conda_envs\pt312` 环境下，已确认：

- `torch==2.7.1+cu118`
- `torch.cuda.is_available() == True`

## 适用范围

当前项目主要面向：

- 交通流/速度预测
- 时空图预测任务
- 多模型统一对比实验
- 后续接入更多时空模型与数据集
