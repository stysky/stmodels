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

- 原始时序：`data/raw/metr-la.h5`
- 官方图来源：`data/raw/adj_METR-LA.pkl`
- 项目标准图：`data/graphs/metr-la_adj.npz`

### `pems-bay`

- 原始时序：`data/raw/pems-bay.h5`
- 官方图来源：`data/raw/adj_mx_bay.pkl`
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
