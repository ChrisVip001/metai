# MetaI 工程规格与验收

MetaI 以 Rust、Burn 和 CubeCL 为基础，面向长上下文、稀疏 MoE 的训练与推理。当前本地稠密模型采用 64、128、256 MiB 三种 FP32 权重上限。

| 方向 | 仓库实现 | 后续验收 |
| --- | --- | --- |
| 架构 | GQA、MLA、RoPE、SwiGLU、专家路由、KV 块选择与分页缓存 | 百万 token 上下文、混合稀疏注意力、MTP、mHC |
| 训练 | 多后端入口、梯度累积、本地同步数据并行、断点恢复 | 跨节点 DDP、专家 All-to-All、状态分片与通信重叠 |
| 精度与部署 | FP32 权重、INT4 量化入口、独立草稿投机解码 | BF16、FP8、NVFP4 性能验收和长上下文部署 |
| 对齐与决策 | SFT、LoRA、偏好优化、策略更新、校准决策头 | 大规模采样、多轮工具轨迹和分布外评测 |
| 模态 | 视觉与音频前缀、Omni 融合、线性注意力与掩码扩散实验 | 开放域数据、消融和推理性能 |

稀疏注意力的块选择在 `src/model/sparse.rs`，紧凑 KV 读取在 `src/model/sparse_exec.rs`，分页缓存在 `src/infer/cache.rs`。集群容量可先用 `plan-cluster --help` 核算。实际规模化验收需记录设备拓扑、上下文长度、吞吐、峰值内存、精度和故障恢复。

[中文教程](COURSE.zh-CN.md) · [返回首页](../README.md)
