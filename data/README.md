# Data Directory

`data/` 用来集中维护项目运行所需的数据资产，避免模型目录里再散落重复副本。

## 目录结构

```text
data/
|-- README.md
|-- graphs/
|   |-- metr-la_adj.npz
|   |-- pems-bay_adj.npz
|   `-- {dataset}_adj.npz
`-- raw/
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
    |-- PEMS08/
    |   |-- PEMS08.npz
    |   `-- PEMS08.csv
    `-- MyDataset/
        |-- dataset.json
        |-- data.npz
        |-- sensor_ids.txt
        `-- edges.csv
```

## 目录约定

### `raw/`

存放原始时序文件和图结构来源文件。

- 官方时序文件可以是 `h5` 或 `npz`
- 官方图结构来源可以是 `pkl`、稀疏 `npz` 或边表 `csv`
- 自定义数据集通过 `data/raw/**/dataset.json` 描述

注意：

- `.gitignore` 默认忽略 `data/raw/*`
- 仓库里默认只保留 `data/raw/dataset.template.json`
- 真正的原始数据请自行放到对应目录

### `graphs/`

存放项目内部统一使用的标准图结构文件，格式为 `*.npz`。

- 对于 `metr-la` / `pems-bay`，框架会优先把官方 `pkl` 转成这里的标准图文件
- 对于 `pems03` / `pems04` / `pems07` / `pems08`，框架会在需要时根据官方 `csv` 边表生成标准图文件
- 如果自定义数据集配置了 `graph_path`，也会优先把规范化后的图写回这里

## 当前内置数据集

| 数据集 | 时序文件 | 图结构来源 | 标准图输出 |
| --- | --- | --- | --- |
| `metr-la` | `data/raw/METR-LA/metr-la.h5` | `data/raw/METR-LA/adj_METR-LA.pkl` | `data/graphs/metr-la_adj.npz` |
| `pems-bay` | `data/raw/PEMS-BAY/pems-bay.h5` | `data/raw/PEMS-BAY/adj_mx_bay.pkl` | `data/graphs/pems-bay_adj.npz` |
| `pems03` | `data/raw/PEMS03/PEMS03.npz` | `data/raw/PEMS03/PEMS03.csv` | `data/graphs/pems03_adj.npz` |
| `pems04` | `data/raw/PEMS04/PEMS04.npz` | `data/raw/PEMS04/PEMS04.csv` | `data/graphs/pems04_adj.npz` |
| `pems07` | `data/raw/PEMS07/PEMS07.npz` | `data/raw/PEMS07/PEMS07.csv` | `data/graphs/pems07_adj.npz` |
| `pems08` | `data/raw/PEMS08/PEMS08.npz` | `data/raw/PEMS08/PEMS08.csv` | `data/graphs/pems08_adj.npz` |

补充说明：

- `pems03` 还会读取 `data/raw/PEMS03/PEMS03.txt` 作为传感器 ID
- `pems04` / `pems08` 默认是 3 维特征：`flow / occupancy / speed`
- `metr-la` / `pems-bay` / `pems03` / `pems07` 当前默认目标特征是单通道 `flow` 或 `speed`

## 图结构解析策略

`spatiotemporal/datasets.py` 中的数据集加载器支持以下 `adjacency_strategy`：

- `auto`
  按顺序尝试：官方 `pkl` -> 标准图 `graph_path` -> 备用 `graph_source_npz_path` -> 官方 `csv` -> 相关性建图
- `official` / `official_pkl`
  优先读取官方 `pkl`
- `official_csv` / `edge_csv`
  根据边表 `csv` 构图
- `canonical` / `graph_npz`
  直接读取 `graph_path` 指向的标准图
- `fallback_npz`
  读取 `graph_source_npz_path`
- `correlation`
  根据时序相关性即时建图
- `none`
  不加载图结构

默认配置来自 [configs/_base/common.json](/e:/Project%20Py/Pinjian_spatialTemporal/configs/_base/common.json:1)：

```json
{
  "dataset_kwargs": {
    "adjacency_strategy": "auto",
    "zero_as_missing": true,
    "fill_method": "interpolate"
  }
}
```

如果使用 `correlation`，还可以额外传：

- `correlation_top_k`
- `correlation_min_weight`

## 数据清洗约定

当前默认流程会：

- 把目标特征中的 `0` 视作缺失值
- 先做时间方向插值 `interpolate`
- 再执行前向和后向补齐

可通过 `dataset_kwargs` 或自定义 `dataset.json` 覆盖：

- `zero_as_missing`
- `fill_method`
- `clean_feature_indices`

其中 `fill_method` 目前支持：

- `interpolate`
- `ffill`
- `bfill`

## 自定义数据集快速开始

框架除了内置 `metr-la / pems-bay / pems03 / pems04 / pems07 / pems08`，也支持自动发现自定义数据集。

推荐目录结构：

```text
data/raw/MyDataset/
|-- dataset.json
|-- data.npz
|-- sensor_ids.txt          # 可选
`-- edges.csv               # 可选
```

### 1. 复制模板

把 [dataset.template.json](/e:/Project%20Py/Pinjian_spatialTemporal/data/raw/dataset.template.json:1) 复制到 `data/raw/MyDataset/dataset.json`。

### 2. 填写最小必要字段

至少需要：

- `name`
- `data_format`
- `data_path`

常用字段：

- `data_key`
- `default_start_date`
- `default_freq`
- `feature_names`
- `target_feature_name`
- `clean_feature_indices`
- `zero_as_missing`
- `fill_method`

### 3. 如果有图结构，填写图相关字段

支持以下来源：

- `graph_source_pkl_path`
- `graph_source_npz_path`
- `graph_source_csv_path`

标准图输出位置可通过以下字段指定：

- `graph_path`

如果是边表 `csv`，还可以继续配置：

- `graph_source_col`
- `graph_target_col`
- `graph_weight_col`
- `graph_weight_mode`
- `graph_threshold`
- `graph_include_reverse`
- `graph_self_loop_weight`

### 4. 如果没有图结构

运行时把 `dataset_kwargs.adjacency_strategy` 设成：

- `correlation`
- `none`

### 5. 直接运行

```bash
python scripts/run_experiment.py --model dcrnn --dataset my-dataset
```

前提是 `data/raw/MyDataset/dataset.json` 里的 `name` 填的是 `my-dataset`。

## 自定义数据格式说明

### 时序文件

- `h5`
- `npz`

### `npz` 约定

- 默认读取键名 `data`
- 数据形状应为 `[T, N]` 或 `[T, N, F]`
- 如果是多特征输入，`feature_names` 数量必须和 `F` 一致
- `target_feature_name` 指定训练时要预测的目标通道

### 传感器 ID

- 可通过 `sensor_ids_path` 指向 `txt` 或 `csv`
- 若未提供，框架会默认生成 `0..N-1`
- 如果图结构文件里的节点标识不是顺序编号，建议显式提供

### 边表 `csv` 约定

- 默认源列名是 `from`
- 默认目标列名是 `to`
- 权重列可以是 `distance`、`cost` 或你自定义的列名
- 没有权重列时也可以按无权图处理

## 维护建议

- 原始数据统一放在 `data/raw/`
- 项目内部稳定复用的图文件统一放在 `data/graphs/`
- 不要再把数据副本散落到模型目录或 `runs/` 目录
- 新增数据集时，优先用 `dataset.json` 接入；只有在确实需要内置注册时再修改 [spatiotemporal/datasets.py](/e:/Project%20Py/Pinjian_spatialTemporal/spatiotemporal/datasets.py:1)
