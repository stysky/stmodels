# 数据目录说明

- `raw/`
  - 存放原始时序数据和原始图结构资产
- `graphs/`
  - 存放当前统一框架内部使用的标准图结构文件

说明：

- `data/raw/adj_METR-LA.pkl`
  - 作为 `METR-LA` 的官方图来源
- `data/graphs/metr-la_adj.npz`
  - 作为项目内部统一使用的标准图文件
  - 由官方 `pkl` 转换得到
- `data/graphs/pems-bay_adj.npz`
  - 当前作为 `PEMS-BAY` 的标准图文件
  - 由官方 `data/raw/adj_mx_bay.pkl` 转换得到

以前分散在各模型目录中的重复数据副本已经移除，现在数据资产统一集中在这个目录下维护。
