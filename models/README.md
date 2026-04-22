# 模型目录说明

这个目录现在只保留统一框架仍然直接依赖的模型骨架源码。

- `Graph-WaveNet-master/`
  - 保留：`model.py`、`LICENSE`
- `stgcn-main/`
  - 保留：`model/`、`LICENSE`

原来模型仓库中的独立训练脚本、测试脚本、重复数据、旧权重、图示文件和缓存已经移除。

现在统一使用：

- `scripts/run_experiment.py`
- `configs/`
- `spatiotemporal/`

来完成训练、导出和推理流程。
