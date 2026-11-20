# Transformer 算子演进时间线（存档 · 2017 → 2026-09）

> 本文保留早期架构设计和当时的判断。当前实现与实验状态以[课程教程](COURSE.zh-CN.md)及[本机验收记录](CPU_RUN.zh-CN.md)为准。


> 本文件是 `metai` 算子级改造的**依据性存档**：按算子族整理 Transformer 各模块历年升级，
> 标注时间、代表模型与「权重/结构影响」，并映射到本仓库的算子文件与落地状态。
> 整理截止：2026-09。信息来源：arXiv、各模型 Technical Report / Model Card、社区源码解析（交叉核对）。

## 0. 与本仓库的算子映射

本仓库把 Transformer 拆为以下算子族（`src/model/`）：

| 族 | 仓库文件 | 角色 | 权重 |
|---|---|---|---|
| 归一化 | `norm.rs` | Pre-Norm 归一化 | gamma 等 |
| 位置编码 | `positional_encoding.rs` | RoPE | 频率表（非参数） |
| 多头布局 | `heads.rs` | split/merge/repeat_kv | 无 |
| 注意力核心 | `sdpa.rs` | 缩放点积 + mask | 无 |
| KV 稀疏（KV 选择） | `sparse.rs` / `sparse_exec.rs` | 位置打分 + Top-k 块 + 滑动窗 | 无（mask 位 / 紧凑 gather 执行位） |
| 注意力组合 | `attention.rs` | GQA 编排 | w_q/k/v/o |
| 前馈激活 | `ffn.rs` | SwiGLU | w1/w2/w3 |
| 路由 | `router.rs` | Top-K 决策 | 无（gate 在 MoE） |
| 稀疏专家 | `moe.rs` | MoE 编排 | gate + 专家 |
| 残差 | `residual.rs` | 残差捷径 | 无 |
| 层/模型组合 | `mod.rs` | TransformerBlock / MetaIModel | — |

---

## 1. 归一化族（norm）

| 时间 | 变体 | 代表 | 要点 | 权重/结构影响 |
|---|---|---|---|---|
| 2015 | BatchNorm | Ioffe & Szegedy | 依赖 batch 统计，语言模型弃用 | — |
| 2016 | **LayerNorm** | BERT / GPT-2 | mean/var + γ/β | 参数级 |
| 2019 | **RMSNorm**（现状默认） | LLaMA 系 | 仅二阶矩 + γ | 参数级 |
| 2025-03 | **DyT（Dynamic Tanh）** | CVPR 2025 / Meta BLT | `tanh(αx)` 逐元素替换归一化，α 可学，可去归一化 | 参数级 |
| 2025-09 | **Zero-Centered RMSNorm** | Qwen3-Next | 先减均值再 RMS，稳定门控线性注意力 | 参数级 |
| 2026-09 基线 | RMSNorm / QK-RMSNorm 仍主流；DyT 试水；无范式级替代 | | | |

## 2. 残差 / 层连接结构族（residual）

| 时间 | 变体 | 代表 | 要点 | 影响 |
|---|---|---|---|---|
| 2017 | Post-Norm | Vaswani | 原始 | 结构级 |
| 2019-2020 | **Pre-Norm**（现状默认） | GPT-2/3 定型 | 现代默认 | 结构级 |
| 2020 | ReZero | Bachlechner | 可学标量门控残差 | 参数级 |
| 2021 | LayerScale | CaiT | 逐通道残差缩放 | 参数级 |
| 2022 | DeepNorm | DeepNet | 系数依赖层数的 post-norm | 参数级 |
| 2023-2026 | 结构固化 | 各厂 | pre-norm + norm-last；门控语义被线性注意力部分吸收 | — |

## 3. 位置编码族（positional_encoding）

| 时间 | 变体 | 代表 | 要点 | 影响 |
|---|---|---|---|---|
| 2017 | 正弦 / 可学习绝对 | Vaswani / GPT | | 无参数 / 参数级 |
| 2019-2020 | 相对 bias / 段 | T5 / Transformer-XL | bias 表 | 参数级 |
| 2021-2022 | **ALiBi** / **RoPE**（现状默认） | Mosaic / NeoX、LLaMA | 斜函数 / 旋转频率 | 无参数 |
| 2023 | RoPE 长文外推 | PI / NTK / **YaRN** / Dynamic NTK | 仅改频率 | 无参数 |
| 2024-10 | CoPE | Meta（论文级） | 上下文位置经注意力门控 | — |
| 2025-04 | **iRoPE / 交错注意力（可无位置嵌入）** | Llama 4 | 支撑 10M 上下文 | 结构级 |
| 2025-09 | **Partial RoPE** | Qwen3-Next | 线性注意力层仅部分维度旋转 | 无参数 |
| 2024-2026 | **纯线性/SSM 层无位置编码** | Mamba 系 | 卷积/状态自带位置感 | 结构级 |
| 2026-09 基线 | 长上下文不押注 PE 外推，改走稀疏注意力 + 检索 | | | |

## 4. 注意力族（attention / heads / sdpa）

### 4.1 头共享/压缩（heads layout）

| 时间 | 变体 | 代表 | 要点 |
|---|---|---|---|
| 2017 | MHA | Vaswani | — |
| 2019 | **MQA** | Shazeer | 全共享 KV（n_kv=1 即 MQA） |
| 2023 | **GQA**（现状默认） | Ainslie，LLaMA2/Mistral | 分组 KV |
| 2024-05 | **MLA** | DeepSeek-V2/V3 | 低秩 KV 压缩 + 解耦 RoPE |
| 2024-10 | **Diff Attention** | Differential Transformer | 双流 softmax 相减 + 可学 λ |
| 2025-09 | MLA × 稀疏 | DeepSeek-V3.2 | 稀疏场景下 MLA 变体 |
| 2026-02 | **MLA 结构升级**（head_dim 192→256，头数 −1/3） | GLM-5 | 降解码算力 |

### 4.2 数值/局部性变体（sdpa 内）

| 时间 | 变体 | 代表 |
|---|---|---|
| 2023 | Sliding-window 局部注意 | Mistral |
| 2024-12 | **QK-RMSNorm** | DeepSeek-V3 |
| 2025-04 | 推理时 attention logits 温度缩放 | Llama 4 |

### 4.3 注意力算法族（2024H2 后主战场）

| 时间 | 变体 | 代表 | 要点 |
|---|---|---|---|
| 2022-2024 | FlashAttention-1/2/**3** | Dao | FA3 = Hopper FP8/TMA |
| 2023-2024 | **PagedAttention** / FlashInfer / prefix-cache | vLLM | 分页 KV（本仓库 `infer/cache.rs` 对标） |
| 2024-06 | FlexAttention | PyTorch | 可编程 mask/稀疏 |
| 2023-2025 | 线性注意力谱系 | RetNet / Mamba / Mamba-2(SSD) / DeltaNet / Gated DeltaNet / RWKV / HGRN / Gated Attention | 隐状态替代 KV |
| 2025-02 | **NSA** | DeepSeek | 粗粒度压缩 + 细粒度选择 + 滑动窗 |
| 2025-02 | **MoBA** | Kimi/清华/浙大 | KV 块级 MoE 路由 |
| 2025-09 | **混合注意力**（Gated DeltaNet×3 + Gated Attention） | Qwen3-Next | 长文 + 高性价比 |
| 2025-09/12 | **DSA**（Lightning Indexer + Top-k + FP8） | DeepSeek-V3.2 | 训练/推理一致的稀疏注意力 |
| 2025-12 | **Titans** | Google（NeurIPS 2025） | 神经长期记忆组件 |
| 2025-12 | **hybrid Mamba-Transformer MoE** | NVIDIA Nemotron 3 | NVFP4 |
| 2026-02 | 稀疏注意力量产（DSA 路线 + block-sparse） | GLM-5 | 2026 旗舰标配 |
| 2026-09 基线 | 旗舰 = MoE（FFN 稀疏）+ 稀疏注意力（KV 稀疏）双维；闭源（Gemini 3 等）细节未公开 | | |

## 5. FFN / 激活族（ffn）

| 时间 | 变体 | 代表 |
|---|---|---|
| 2017 | ReLU；GLU（Dauphin） | 原始 |
| 2019-2020 | **GELU / SiLU**；GEGLU | BERT/GPT-2 |
| 2022 | **SwiGLU**（现状默认） | PaLM → LLaMA 系定标 |
| 2024-09 | Squared ReLU | 论文级 |
| 2025-2026 | 无新范式激活；主流仍 SwiGLU/GeGLU；FFN 转为 MoE 专家网络 | Qwen3/DS/GLM 全系 |

## 6. MoE 路由与负载均衡族（router / moe）

| 时间 | 变体 | 代表 | 要点 |
|---|---|---|---|
| 2017 | Top-2 + noisy gating | Shazeer | 起源 |
| 2020-2021 | GShard（capacity+aux）→ Switch（Top-1） | Google | 定范式 |
| 2022 | ST-MoE | Google | z-loss、分层 aux |
| 2023-2024 | Top-k 量产 | Mixtral 等 | — |
| 2024-05 | **细粒度专家 + 共享专家**（现状支持） | DeepSeek-V2 | 共享吸收公共模式 |
| 2024-12 | **sigmoid 无 softmax 路由 + bias 均衡（无 aux loss）** | DeepSeek-V3 | 数值稳定、训练友好 |
| 2025-04 | 共享专家 + 大规模稀疏（16/128 专家） | Llama 4 | MoE 首次进 Llama |
| 2025 | shared experts 延续 | Qwen3-235B | — |
| 2025-09/12 | 线性层也 MoE 化 / 极致稀疏（3B active） | Qwen3-Next / Nemotron 3 | MoE × 线性注意力协同 |
| 2026-02 | DSA + MoE + MTP | GLM-5 | 双维稀疏集齐 |
| 2026-05 | 格局成型：排行 top6 全部 MoE | 业界 | 旗舰 Dense 消亡 |

## 7. Embedding / 输出 / 训练-推理接口族

| 时间 | 变体 | 代表 |
|---|---|---|
| 2019 | tied/untied embedding、缩放 | GPT-2 |
| 2024-12 | **MTP 多 token 预测模块** | DeepSeek-V3 → GLM-5 |
| 2025-12 | **token 压缩**（Lightning Indexer 摘要） | DeepSeek-V3.2 |

## 8. 相关训练侧（非算子，但影响算子接口）

| 时间 | 内容 |
|---|---|
| 2024-12 | FP8 训练（DeepSeek-V3） |
| 2025-12 | NVFP4（Nemotron 3） |
| 2026 | **Muon 优化器**在万亿 MoE 量产（正交化友好需投影结构预留，GLM-5 为其拆分 MLA 投影） |

---

## 9. 落地状态（仓库实现 vs 变体清单）

状态图例：🟢 已落地 · 🟡 参数级变体易落地 · 🔴 结构/算法级需新抽象

| 族 | 现状默认 | 已具变体 | 待落地优先级 |
|---|---|---|---|
| 归一化 | RMSNorm | — | 🟡 LayerNorm / DyT / Zero-Centered RMSNorm |
| 残差 | Pre-Norm add | — | 🟡 结构开关（Post/Pre） |
| 位置编码 | RoPE | RoPE 外推（频率预留） | 🟡 Partial RoPE；🟡 YaRN/NTK；🔴 无 PE（线性层） |
| 头布局 | GQA | MHA/MQA（n_kv 特例） | 🔴 MLA；🔴 Diff Attention |
| 注意力核心 | naive SDPA | — | 🟢 **稀疏已三位落地**：① mask 语义位（`config.sparse`，训练/推理同构）；② 解码紧凑 gather 执行位（`sparse_exec.rs`，只对被选块∪窗口打分，O(K)→O(L)，与 mask 位逐位等价有测试）；③ **cache 页读取**（`cache.gather_spans` + `head_rows`，块与页对齐时消除每步 `get_full` 的 O(K) 拼接，解码直接以 cache 为数据源）；🟡 对齐要求 `block_size==16`，否则自动回退；🔴 cubecl 单 kernel 融合（零 host 同步）；🔴 Flash kernel |
| 前馈 | SwiGLU | — | 🟡 GeGLU / GELU 变体 |
| 路由 | softmax Top-K | — | 🟡 sigmoid Top-K（DS-V3 风格，无 aux loss） |
| MoE | 路由专家+共享专家 | shared 开关 | 🟡 细粒度专家 |
| 输出 | Linear | — | 🔴 MTP 模块 |

## 10. 演进判断（截至 2026-09）

1. **两轮稀疏化是主线**：FFN 侧 MoE 已全面铺开（2026-05 顶级模型全 MoE）；KV 侧重心转向稀疏/选择型注意力（NSA→MoBA→DSA→GLM-5）。
2. **混合线性注意力**（Qwen3-Next / Nemotron 3）是 Transformer 之外的现实竞争者，主流仍是「Transformer 主干 + 局部轻量层」。
3. 归一化/激活/残差进入平稳期；创新集中在**注意力形态**与**推理期 KV 管理**。
4. 对 metai 的意义：接口上必须让「注意力核心」「路由」「归一化」成为可组合的替换点，未来两年新增多为这三族。
