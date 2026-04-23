# Data Directory

`data/` 用来集中维护项目运行所需的数据资产，避免模型目录里再散落重复副本。

## 目录结构

### `raw/`

存放原始时序文件和官方图结构来源文件：

- `metr-la.h5`
- `pems-bay.h5`
- `adj_METR-LA.pkl`
- `adj_mx_bay.pkl`

这些文件是数据加载的源头，不建议在训练流程里直接修改。

### `graphs/`

存放项目内部统一使用的标准图结构文件：

- `metr-la_adj.npz`
- `pems-bay_adj.npz`

框架优先使用这里的规范化图文件；如果只存在官方 `pkl`，数据集加载器也会在需要时自动转换并写回这里。

## 当前数据集映射

### `metr-la`

- 原始时序：`data/raw/METR-LA/metr-la.h5`
- 官方图来源：`data/raw/METR-LA/adj_METR-LA.pkl`
- 项目标准图：`data/graphs/metr-la_adj.npz`

### `pems-bay`

- 原始时序：`data/raw/PEMS-BAY/pems-bay.h5`
- 官方图来源：`data/raw/PEMS-BAY/adj_mx_bay.pkl`
- 项目标准图：`data/graphs/pems-bay_adj.npz`

## 图结构解析策略

`spatiotemporal/datasets.py` 中的数据集加载器支持多种 `adjacency_strategy`：

- `auto`：优先官方图，再落到项目标准图，最后可退化为相关性建图
- `official` / `official_pkl`：优先使用官方 `pkl`
- `canonical` / `graph_npz`：优先使用 `data/graphs/*.npz`
- `fallback_npz`：从备用 `npz` 读取
- `correlation`：根据时序相关性即时建图
- `none`：不加载图结构

默认配置使用的是：

```json
{
  "adjacency_strategy": "auto",
  "zero_as_missing": true,
  "fill_method": "interpolate"
}
```

## 数据清洗约定

当前默认流程会：

- 把速度值 `0` 视作缺失值
- 对缺失值做时间方向插值
- 在插值后再执行前向 / 后向补齐

如果你想保留原始 `0` 值，或者切换为别的补齐方式，可以在 `dataset_kwargs` 里覆盖：

- `zero_as_missing`
- `fill_method`

## 维护建议

- 原始文件统一放 `data/raw/`
- 项目内部稳定使用的图文件统一放 `data/graphs/`
- 不要再把图文件或数据副本放到模型目录里
- 新增数据集时，优先在 `spatiotemporal/datasets.py` 里登记输入文件路径和图结构来源

## Custom Dataset Quick Start

现在框架除了内置 `metr-la / pems-bay / pems03 / pems04 / pems07 / pems08`，也支持自动发现自定义数据集。

推荐目录结构：

```text
data/
|-- graphs/
`-- raw/
    |-- dataset.template.json
    `-- MyDataset/
        |-- dataset.json
        |-- data.npz
        |-- sensor_ids.txt          # 可选
        `-- edges.csv               # 可选
```

使用方式：

1. 把 [dataset.template.json](/e:/Project%20Py/Pinjian_spatialTemporal/data/raw/dataset.template.json) 复制成 `data/raw/MyDataset/dataset.json`
2. 根据你的数据格式填写 `data_format / data_path / feature_names / target_feature_name`
3. 如果有图结构：
   填 `graph_source_pkl_path`、`graph_source_npz_path` 或 `graph_source_csv_path`
4. 如果没有图结构：
   在运行配置里把 `dataset_kwargs.adjacency_strategy` 设成 `correlation` 或 `none`

当前支持的常见输入格式：

- 时序数据：`h5`、`npz`
- 传感器列表：`txt`、`csv`
- 图结构：官方 `pkl`、稀疏 `npz`、边表 `csv`

`npz` 约定：

- 主数据默认读取键名 `data`
- 数据形状应为 `[T, N]` 或 `[T, N, F]`
- 如果是多特征输入，`feature_names` 数量必须和 `F` 一致
- `target_feature_name` 指定训练要预测的目标通道

边表 `csv` 约定：

- 默认列名为 `from` / `to`
- 权重列可以是 `distance` 或 `cost`
- 没有权重列时也可以按无权图处理

如果只想直接跑你自己的数据集，不改源码也可以：

```bash
python scripts/run_experiment.py --model dcrnn --dataset my-dataset
```

前提是 `data/raw/<你的目录>/dataset.json` 里的 `name` 填的是 `my-dataset`。
