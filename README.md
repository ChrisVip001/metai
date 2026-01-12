# MetaI (Meta Intelligence)

<div align="center">

**基于 Rust + Burn 0.19.1 的工业级高性能大语言模型 (LLM) 全栈框架**

[核心特性](#✨-核心特性) | [分布式深度解析](#🌐-分布式集群训练-deep-dive) | [快速开始](#🚀-快速开始) | [架构设计](#🏗️-项目架构设计)

---

MetaI 并非一个简单的 LLM 演示项目，而是一个旨在挑战 2026 年大模型工业界巅峰性能的 **全链路研发框架**。我们利用 Rust 的极致安全与并发特性，结合 [Burn](https://github.com/tracel-ai/burn) 深度学习引擎，实现了从单机 125M 原型验证到集群级 **685B 顶配架构** 的无缝切换。

</div>

## ✨ 核心特性：突破技术天花板

### 1. 巅峰性能与工程化
*   **Rust 原生执行**：摒弃复杂的 C++ 绑定，全链路 Rust 实现，确保显存管理的绝对精确。
*   **分布式集群训练**：内置多机多卡 DDP (Distributed Data Parallel) 与专家并行 (Expert Parallel) 模式，原生对抗 NCCL 通信瓶颈。
*   **685B 级架构支持**：率先支持 **DeepSeek-V3.2** 规格，集成 **Multi-Token Prediction (MTP)** 并行训练加速技术。
*   **KV Cache 2.0**: 配合 **Multi-head Latent Attention (MLA)**，在 128K 超长上下文生成时，KV Cache 显存占用仅为传统架构的 1/8。

### 2. 下一代对齐算法 (Alignment)
*   **GRPO (Grouped Relative Policy Optimization)**：原生实现 DeepSeek-R1 同款算法。通过组内相对奖励归一化，无需昂贵的 Critic 模型即可实现强化学习，极大地提升了模型的逻辑推理 (System 2) 能力。
*   **SFT & DPO**：支持带 Loss Masking 的指令微调及直接偏好优化，完美解决变长序列下的 Loss 污染问题。

### 3. 先进推理技术
*   **规格预测解码 (Speculative Decoding)**：支持 Draft Model 引导的快速生成，利用小模型验证大模型 Token，生成速度提升 2x-3x。
*   **INT4 量化部署**：基于 Block-wise 的 4-bit 权重量化，支持在消费级显卡上运行原本无法承载的巨型号模型。

---

## 🌐 分布式集群训练 Deep Dive

在 `metai` 中，大规模分布式训练不再是黑箱。利用 `LibTorch` (Burn-tch) 后端，深度调用 Tensor Cores 算力，通过 `train-cluster` 命令，你可以直接操作 **685B MoE** 级模型：

```bash
# 在 A100/H100 集群上启动 685B 顶配训练
# 需确保系统已安装 CUDA 和 cuDNN
cargo run --release --no-default-features --features cuda -- train-cluster \
    --world-size 128 \
    --chinese-path /mnt/data/huge_zh_corpus.jsonl \
    --english-path /mnt/data/huge_en_corpus.jsonl
```

### 核心优化逻辑：
1.  **专家并行 (Expert Parallelism)**：针对 256 位专家，仅激活 Top-8。系统会自动将不同专家分发至不同节点，利用 All-to-All 通信实现超大规模 MoE。
2.  **吞吐量屏蔽 (Throughput Hiding)**：默认设置 `grads_accumulation = 256`。通过增大累积步数，让 GPU 在进行梯度计算时能同时发起跨机通信，最大化带宽利用率。
3.  **128K 分片加载**：`LazyTextDataset` 会根据设备的 `Rank` 自动进行物理索引切片，确保没有任何一台机器需要预加载全量数据集。

---

## � 快速开始

### 1. 环境准备
- **Rust**: Nightly (推荐) 或 1.70+
- **硬件**: 推荐 Nvidia RTX 30/40 系列 (单机验证) 或 A100/H100/H200 (分布式)
- **驱动**: 需正确配置 CUDA 12.x 或 WGPU 运行时

### 2. 训练你的第一个 Small 模型 (125M)
适合在本地 16GB+ 显存设备上快速运行：

```bash
# 克隆并编译
git clone https://github.com/ChrisVip001/metai.git && cd metai
cargo build --release

# 启动训练 (自动开启 TUI 监控)
cargo run --release -- train-small \
    --chinese-path 4in1.txt \
    --english-path dataset.txt \
    --output-dir output_small
```

---

## 🧠 RLHF 强化学习与推理增强 (GRPO)

MetaI 深度集成了推理增强技术，旨在让模型学会“思考”：

```bash
# 针对数学、代码推理任务进行 GRPO 训练
cargo run --release -- train-grpo \
    --data data/math_reasoning.jsonl \
    --model-dir checkpoints/sft_warmup \
    --output-dir checkpoints/grpo_reasoning
```

**算法优势：**
- **Advantage Normalization**: 在每组 sample（Group Size 可配置）内部计算相对奖励，强制模型在组内竞争。
- **Kullback–Leibler (KL) Penalty**: 自动控制 Policy 偏离 Reference 模型的程度，确保训练稳定性。

---

## 🧪 模型配置深度对比

| 配置标识 | 总参数量 | 特色架构 | 上下文窗口 | 典型硬件 |
|:---|:---|:---|:---|:---|
| **Small** | 125M | 8E MoE, Top-2 | 2K | RTX 3060 (12G) |
| **DS-V3.2** | **685B** | **256E MoE, MLA, MTP** | **128K** | **H100 x 8 Cluster** |

---

## 🏗️ 项目架构设计

```text
src/
├── model/
│   ├── attention.rs         # 集成 MLA (Multi-head Latent Attention) 与 GQA
│   ├── mlp.rs               # 稀疏 MoE 专家层，支持动态专家动态负载均衡
│   ├── config.rs            # 从 Llama 到 DeepSeek-V3.2 的全量配置定义
│   └── positional_encoding.rs # RoPE 旋转位置编码 (支持长文本外推)
├── train/
│   ├── distributed.rs       # 分布式 DDP 与跨机负载均衡逻辑
│   ├── grpo.rs              # 强化学习算法实现 (不需要 Value Model)
│   ├── sft.rs               # 监督微调，带 Loss Masking 机制
│   └── mod.rs               # 统一训练入口与 Learner 封装
├── infer/
│   ├── cache.rs             # 工业级 KV Cache 管理 (支持 MLA 压缩)
│   └── mod.rs               # 文本生成引擎与投机采样器
└── data/
    └── data.rs              # 基于磁盘索引的 Lazy Loading，支持 TB 级数据
```

---

## 🤝 贡献与加入

MetaI 致力于建立 Rust 在大语言模型领域的 **顶级生态**。如果你对分布式系统设计、GPU 算子优化或尖端对齐策略感兴趣，欢迎随时开启 Pull Request 或 Issue。

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

Made with ❤️ by the MetaI Team | (c) 2026 Meta Intelligence

</div>
