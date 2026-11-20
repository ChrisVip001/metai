# MetaI 工业规格与实机记录

MetaI 的交付目标是覆盖单机 125M 原型、长上下文、万亿级稀疏模型的训练、对齐和部署。CPU 回归用于尽早发现算子、梯度同步、数据泄漏和恢复错误，中文课程用于传播同一套实现。缩小回归模型无须缩小产品目标。变更目标规模、删换公共训练入口、切换主后端或改变 checkpoint 兼容策略前，维护者先说明原因、影响及迁移方案，与项目负责人确认。

## 单机 125M 双语训练

当前 125M 复验使用完整 16K 字节级词表和 8,192 个预训练句对。本轮的单机训练记录、56 道新题与 SFT 对照见[第二版实机实验](M2_BILINGUAL_V2.zh-CN.md)。

第一版在 M2 Pro Metal 上从随机权重预训练 50,682 个有效 token，再用 256 条真实中英文指令做两轮 SFT。两次正式训练合计 192 次参数更新。语料与署名、两份权重在 40 道留出题上的逐条回答、分词器对数字的表达限制见[真实语料训练与评测](M2_BILINGUAL.zh-CN.md)。下方保留最初两步硬件验收的记录。

## 单机 125M 工程原型

| 项目 | 2026-11-20 本机记录 |
| --- | --- |
| 硬件、后端 | Apple M2 Pro、16 核集成 GPU、Burn 0.22.1、CubeCL Metal，调试构建 |
| 实际结构 | 125,276,928 个参数，16 层，隐藏维 768，12 个查询头、4 个 KV 头，SwiGLU 宽度 2048，每层 1 个路由专家，16K 词表，上下文上限 2048 |
| 实际工作量 | 单样本、一次 FP32 AdamW 更新，训练批损失 10.109167。独立文本的 5 个目标 token 验证 NLL 为 10.293401 |
| 模型与优化器 | `model.bpk` 501,123,840 字节，`optimizer.bpk` 1,002,270,976 字节，同时保存配置、分词器和训练进度 |
| 分配器 | 验证后、保存前采样，使用 1,503,610,288 字节，保留 1,518,253,568 字节。这是主设备 Burn 分配器在该时刻的读数，整机统一内存峰值需单独测量 |
| 重载 | 另一个进程从 BurnPack 重载，在同一 Metal GPU 上从 `The sky` 贪心生成四个新 token。解码后的整句为 `The sky is blue.`，训练文本本身含有相同续写，无法由此推断开放域语言能力 |

下面是同一工作负载。用新目录保留每次运行，`model.bpk` 和优化器合计约 1.40 GiB。使用仓库自带的 `tokenizer.json` 可复现参数规模。重新拟合 BPE 时，预训练会按实际词表大小分配嵌入与输出层。

```bash
RUN=$(mktemp -d)
printf '%s\n' 'the sky is blue.' > "$RUN/train.txt"
printf '%s\n' 'red is a color.' > "$RUN/valid.txt"
cargo run --locked --no-default-features --features metal -- \
  --device metal pretrain --size 125m --tokenizer-path tokenizer.json \
  --train-path "$RUN/train.txt" --valid-path "$RUN/valid.txt" \
  --batch-size 1 --num-epochs 1 --output-dir "$RUN/125m"
cargo run --locked --no-default-features --features metal -- \
  --device metal generate --model-dir "$RUN/125m" --prompt 'The sky' --max-len 4
cargo run --locked --no-default-features --features metal -- \
  --device metal resume-pretrain --model-dir "$RUN/125m" \
  --train-path "$RUN/train.txt" --valid-path "$RUN/valid.txt" \
  --additional-epochs 1 --output-dir "$RUN/125m-resumed"
```

从上述 125M 断点恢复后，第二次 AdamW 更新在同一训练样本上的损失为 1.295044，独立文本验证 NLL 升到 10.578198。这一对读数说明了为什么跑通 GPU 训练和取得可部署的语言质量需要不同的证据。[第二步原始运行报告](experiments/metal-125m-resumed.json)

新运行将 `training-report.json` 与最终权重一同发布，记录设备后端、参数量、实际全局批量、训练块及目标 token 数、验证 NLL 和主设备分配器快照。无分配器统计的后端该字段为 `null`。权重、AdamW 状态、语料指纹及进度可用于 epoch 边界恢复。当前这份 125M 权重可用于加载、推理、性能分析或继续训练。面向用户交付的模型需要有代表性的授权数据、独立任务评测和延迟、成本、错误率报告。

## 设备与分布式执行

Burn 的设备由运行时选择。用 Cargo feature 编译所需后端，通过全局 `--device` 从本二进制包含的后端中选择。项目 `cuda` 使用 `burn/cuda` 和 `Device::cuda(rank)`，Metal 使用 `burn/metal`。Burn 0.22.1 上游仍提供弃用的可选 `tch`，MetaI 的 CUDA 正常依赖树通过 `burn-cuda`。项目提供 ROCm、Vulkan、WebGPU、WGPU 与 CubeCL CPU 特性入口，单个入口的存在不能替代对应硬件测试。默认 `cpu` 特性使用 Flex，便于普通机器执行 CI。Metal 完成训练和重载，CUDA 已在本机完成全目标编译检查，CUDA 驱动及 NVIDIA 实际内核仍需 NVIDIA 节点验证。[Burn 0.22.1 文档](https://docs.rs/burn/0.22.1/burn/) · [Burn CubeCL 后端](https://docs.rs/crate/burn-cubecl/0.22.1)

模型调用使用 Burn 0.22 的无后端类型参数 API，设备上的自动微分通过 `device.autodiff()` 开启。CubeCL 编译依赖中包含 Pliron，CUDA 构建还会使用上游 LLVM 工具链。此处“纯 Rust 栈”指模型实现及 Burn/CubeCL 计算路径，部署机器仍需相应驱动和编译支持。编译耗时、显存节省和低精度吞吐必须在相同硬件与构建缓存条件下测量后发布。

`train-cluster` 当前实现**单进程、本地设备之间的同步数据并行**。旧 worker 计算过分片起止位置，实际加载器仍接收全量数据，各 worker 独立更新。这次恢复了用户入口，并把执行语义改为以下步骤。

1. `--train-path` 可重复指定。旧的 `--chinese-path` 与 `--english-path` 在集群入口均进入训练，`--valid-path` 始终单列。JSONL 仅读取 `text` 字符串，字段元数据不会进入 BPE 和训练文本。文本内的换行形成新的行边界。
2. 启动时逐行分词，将包含至少一个有效目标的块写入临时磁盘。索引记录偏移、长度和目标 token 数。每次 shuffle 后，具有训练目标的块在每个 epoch 内只分配一次。内存占用包含块索引、全局顺序、当前行和当批张量，磁盘空间随 token 数线性增长。当前分布式实现集中建索引，跨节点物理分片及持久化预分词缓存仍待落地。
3. 一个完整优化步的样本容量等于本地副本数 × 每副本 micro-batch × 累积轮数。每个副本计算本地 token 损失之和，除以本次全局实际非 padding 目标数。各副本的参数梯度移回主设备相加，AdamW 只更新一次，下一步的副本使用更新后的权重。尾批覆盖剩余样本，按实际目标数归约。
4. 这条路径在主设备集中归约梯度，主设备保存整份模型及优化器，每个工作设备也必须容纳完整模型。NVL72 跨机数据并行、EP 的 All-to-All 和切分权重需要独立通信与存储实现。增加 `world-size` 只改变本地副本布局。
5. 为了在 epoch 边界切换拓扑，可为 `resume-pretrain` 指定新的副本数、每副本批量和累积步数。引擎检查全局批量一致，恢复 AdamW 和原始语料指纹，保持输入源的顺序。

```bash
# 原 checkpoint 的全局批量为 8。两份训练源按原顺序传入。
cargo run --locked -- resume-pretrain --model-dir artifacts/epochs/epoch-1 \
  --train-path /data/zh.jsonl --train-path /data/en.jsonl --valid-path /data/heldout.jsonl \
  --world-size 2 --batch-size 2 --accumulation-steps 2 \
  --additional-epochs 1 --output-dir artifacts/resumed
cargo test --locked --test data_parallel -- --nocapture
```

在 CPU 上以逻辑副本检验数值等价性。两份 JSONL、不同长度的样本、非整除尾批、一份独立验证数据。在相同参数和输入上，单副本全局批量与双副本累积训练后的最大 logits 差约为 `3.58e-7`，同拓扑恢复差为 `0`。从普通预训练的同一断点分别单副本续训和切换双副本，差约为 `5.96e-8`。磁盘块读取也与内存读取逐块比较，并覆盖 CRLF、空行和单行跨块。[回归测试](../tests/data_parallel.rs)

## 在没有设备时评审 128 卡方案

`plan-cluster` 从模型总参数、权重位宽及理想的均匀分片数计算裸权重字节数。一个完整模型副本的设备数由 `--weight-shards` 指定，设备总数必须能组成整数个副本。每副本 micro-batch × 副本数 × 累积次数给出全局样本数。再乘序列长度得到理想填满时的 token 位置上限。MoE 每 token 激活的参数影响算量，所有专家的参数仍要存放。

```bash
cargo run --locked -- plan-cluster --model v4-pro --world-size 128 --weight-shards 64 \
  --weight-bits 4 --micro-batch 1 --accumulation-steps 256 --sequence-length 1000000
```

在这组假设下，理想单分片权重占 12,500,000,000 字节，两个副本每次更新共 512 条样本。规划器也接受 `v4-flash` 和仓库 16K 词表的 `125m`。计算结果不会为 EP/TP 分配设备，也不会改变训练器的能力或保证模型放得下。

## 2026 技术目标及其验收

| 子系统 | 仓库已有路径 | 下一项工业验收 |
| --- | --- | --- |
| V4 规格 MoE、长上下文 | GQA、MLA、门控和共享专家、稀疏选择与单步 KV 解码。训练仍包含稠密注意力矩阵 | 实现 CSA/HCA 压缩索引和混合调度、无辅助损失负载均衡、mHC、MTP 和 Muon。用长序列训练和推理分别测 128K 至 1M 的内存、吞吐、检索质量和梯度稳定性 |
| 精度与通信 | 现有同步训练为 FP32，主设备集中归约，原生 checkpoint 为 BurnPack | 多机 DDP、EP/TP 的 dispatch/combine All-to-All、通信与计算重叠、优化器分片。BF16、FP8、NVFP4 分别做硬件内核与精度对照，报告关键高精度层、拓扑、有效带宽和恢复时间 |
| 对齐与 Agent | SFT、DPO、LoRA、PPO、GRPO、GSPO、CISPO、规则/AI 反馈和短工具轨迹 | DAPO 的非对称裁剪、动态采样、token 损失和超长奖励整形；Dr.GRPO、GFPO、ARPO/Tree-GRPO、VAPO、InSPO/TI-DPO 分别与同预算基线比较。长 CoT 和长程工具轨迹需另设独立评测 |
| 推理与部署 | 独立草稿模型投机解码、MLA 缓存、INT4 权重量化入口和蒸馏 | 训练并复用 MTP 头，比较接受率、端到端延迟与显存；实现 QLoRA 训练、分块低精度部署、专家卸载、SafeTensors 导出和 ONNX 等价性验证 |

DeepSeek 发布的 V4-Pro 和 V4-Flash 权重规格分别为 1.6T/49B 与 284B/13B，标注 1M 上下文。它们是 MetaI 的参考规格。DSA 选取的 key 比例属于注意力内核口径，系统还要承担索引、前馈、通信及 KV 读写成本。MLA 的缓存节省需要明确对比的 MHA 头数和精度。MTP 的加速倍数也取决于接受率、批量与硬件。对外发布这些数据时应附 MetaI 实测。以 685B 参数估算，理想裸 4 bit 权重已需 342.5 GB（约 319 GiB），还应加上缩放因子、未量化层、缓存和运行时空间。[DeepSeek V4 官方模型卡与报告](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)

`train-gspo` 已接入原有组内 rollout 和 checkpoint。每条回答在掩码内求新旧策略 log-prob 差的平均，取指数得到序列比率，在序列级裁剪并对回答等权取平均。`--beta 0` 可关闭可选参考策略 KL。当前实现采用可配置的对称裁剪，长轨迹稳定性和相对 GRPO 的质量收益待同预算大样本验证。[GSPO 原论文](https://arxiv.org/abs/2507.18071) · [DAPO 原论文](https://arxiv.org/abs/2503.14476)
