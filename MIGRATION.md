# Migration Notes

本文件记录这轮统一化改造后的结构变化，方便后续继续维护或回看历史差异。

## 1. Adapter 层统一

旧结构中，部分模型有独立 adapter 文件，例如：

- `spatiotemporal/adapters/graph_wavenet.py`
- `spatiotemporal/adapters/stgcn.py`

现在统一为：

- `spatiotemporal/adapters/base.py`
- `spatiotemporal/adapters/unified.py`
- `spatiotemporal/adapters/model_zoo.py`
- `spatiotemporal/adapters/__init__.py`

其中：

- `base.py` 负责抽象基类和公共输入规整
- `unified.py` 负责统一训练 / 导出 / 推理流程
- `model_zoo.py` 负责模型 `Config`、构建函数和注册规格
- `__init__.py` 负责创建入口和配置规整

## 2. 模型注册方式变化

旧方式更偏“每个模型一个 adapter 类文件”。

现在改为：

- `ModelSpec`
- `RegisteredForecastAdapter`
- `MODEL_SPECS`

新增模型时，主要在 `spatiotemporal/adapters/model_zoo.py` 中补充：

1. `Config`
2. `build_model`
3. 必要时补 `build_scheduler`
4. 注册到 `MODEL_SPECS`

## 3. 配置文件支持继承

新增：

- `spatiotemporal/config_loader.py`

配置文件现在支持：

```json
{
  "extends": "_base/common.json",
  "model": "stgcn",
  "dataset": "metr-la"
}
```

共享默认配置放在：

- `configs/_base/common.json`

这样可以减少 `configs/*.json` 的重复内容。

## 4. 运行目录策略

训练输出仍默认写入：

- `runs/{model}/{dataset}/{timestamp-tag}/`

但现在临时验证或冒烟测试建议：

- 使用单独的 `run_root`
- 或通过 API 传 `save_run=False`
- 并显式指定临时 `checkpoint_path`

## 5. Vendor 模型内嵌

原先 `Graph WaveNet` 和 `STGCN` 会从：

- `spatiotemporal/models/vendors/Graph-WaveNet-master/`
- `spatiotemporal/models/vendors/stgcn-main/`

动态加载原仓库代码。

现在这两部分实现已经整理进：

- `spatiotemporal/models/conv_models.py`

因此项目运行不再依赖 `vendors/` 目录。

## 6. 数据目录统一

图结构和原始数据统一集中到：

- `data/raw/`
- `data/graphs/`

不要再把图文件或数据副本放到模型目录中。
