# CPU 小模型实验记录 · 2026-11-20

本页保留 CPU 第一课的历史读数。125M 原型的 GPU 训练、推理和恢复实测见[工业规格与实机记录](INDUSTRIAL.zh-CN.md)。随后针对决策误判和视听乱码完成了单独的留出实验，数据划分、逐条结果和新图见[中文教程第七课](COURSE.zh-CN.md#heldout-comparison)。

## 如何重做这次实验

工作目录为仓库根目录。设备为 arm64 Mac，macOS 26.6.2。Rust 和 Cargo 均为 1.100.0。依赖按 Cargo.lock 解析，Burn 0.22.1，tokenizers 0.23.2。默认计算后端为 Flex，训练开启自动微分。Python 脚本使用标准库。公开 API 和发布版本核对见 [Burn 0.22.1](https://docs.rs/burn/0.22.1/burn/) 和 [tokenizers 0.23.2](https://docs.rs/tokenizers/0.23.2/tokenizers/)。

```bash
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 scripts/run_tiny.py --output artifacts/my-reproduction --seed 42
```

测试结果为 41 个 Rust 测试通过、2 个 Python 测试通过。另有 1 个诊断测试按设计忽略，该测试需手工提供旧实验文本。标准测试覆盖路由梯度、缓存因果性与 padding、响应掩码、LoRA 合并、裁剪边界处的策略梯度、两种感知前缀的梯度、保存重载、预训练恢复及多个训练 CLI 的接力。WGPU 的构建组合另外以 `cargo check --locked --no-default-features --features wgpu --all-targets` 检查。

`run_tiny.py` 执行 26 个 CLI 步骤，脚本会在任一步失败时报告命令和日志。本机留存目录为 `artifacts/cpu-course-2026-11-20/`，`run.json` 中每步都有独立退出码和耗时，`logs/` 保存原始输出。权重和原始日志为本机运行产物，干净检出通过上面的命令重建。

仓库保留了两张可直接对照的图。第一张取数自 `logs/pretrain.log`、`logs/decide.log` 和 `decision/evaluation.json`，并从决策测试集读取正确标签。第二张将视听输入与 `logs/audiovisual-generate.log` 中的解码结果逐行对齐。在装有 Matplotlib 的环境运行 `python3 scripts/plot_results.py artifacts/my-reproduction artifacts/my-reproduction/figures` 即可重画。[预训练与决策图](figures/cpu-training-decision.png) · [视听问答图](figures/multimodal-generation.png)

## 这组数字能检验什么

| 实验 | 实测 | 解释 |
| --- | --- | --- |
| 清洗与切分 | 10 条训练、2 条验证，合并 1 条重复，过滤 1 行空白 | 先去重后切分，清单能追溯原始行号 |
| 语言模型 | V=285，H=16，1 层，2 个专家；3 个 epoch 训练损失 5.735984 → 5.657616 → 5.634100 | 验证共 22 个 token，NLL 为 5.708973。训练在动，语言能力还很弱 |
| epoch 恢复 | 从第 2 个 epoch 恢复后，验证 NLL 同为 5.708973 | 独立集成测试还逐元素比较连续运行和恢复运行的同输入 logits |
| SFT / LoRA / DPO | SFT 第 2 轮训练损失 6.114874；LoRA 训练 1,204 个参数；DPO 首批 0.693147 | 每个分支从指定权重加载，保存后重新从 CLI 读取 |
| PPO / GRPO / CISPO | 同一个初始策略，4 个 prompt 各采样 4 次。规则奖励均值 0.1800 | 这里报告更新前的采样奖励。不同目标在同一首批得到相同奖励属预期，无法据此比较训练以后谁更好 |
| AI 反馈 | 冻结的本地语言模型评分，16 条响应；生成上限改为 2 | 验证可替换奖励来源。该小判断模型没有可靠性基准 |
| 蒸馏、自适应停止 | 学生 H=8，训练 KL 损失 0.317917；停止头在最大预算 4、阈值 0.8 时，四个生成 token 分别执行 3、2、2、2 次隐状态更新 | 需要更强教师和任务级正确率，才能判断性能保留或节省了多少推理计算 |
| 工具和 Agent | 工具模仿头在隐藏卡片示例上实际执行 Lookup，然后正确回答。RL 后贪心策略答对 4 种情形中的 3 种，两种隐藏情形都直接猜 1 | 相同环境可复现“碰巧猜对”。这里的 RL 未稳定学到查询策略 |
| Jev 类决策 | 训练／校准／测试分别 6／4／4 条。测试准确率 2/4，温度 T=8；NLL 从 0.8672 降至 0.7077，多类 Brier 从 0.6483 降至 0.5145 | 此次网格选到温度上界。两类的均匀猜测 NLL 约 0.6931。样本少，不能声称概率已可靠校准。阈值 0.8 下覆盖率为零 |
| 视觉 / Omni | 独立噪声种子的 24 个合成场景上，两种模型各答对 24 个；Omni 两个重建头的 MSE 之和约 0.0234 | 评测沿用相同条纹、音调和三个指令，不覆盖真实图像或语音 |
| 感知前缀语言模型 | 两条图文、四条图像＋音频问答数据各训练一轮，回答 NLL 分别为 5.9360 和 5.6678，重载后均能逐 token 生成 | 目前生成的回答仍含乱码。分类实验的 24/24 与生成实验是两个不同的评测 |
| 线性 LM / 掩码扩散 | 各训练 12 个 epoch，均完整还原四条已见循环模式 | 验证自回归递推和迭代去噪能拟合玩具任务，未评估新语法、新长度或大词表 |

另外用教程中的 MLA 配置完成了一轮小模型训练。配置为 top-1、潜在 KV 宽度 8、RoPE 宽度 4，验证 NLL 为 5.786363。一次单轮运行不足以对两种注意力做质量排名。

## 验证边界

当前记录覆盖单进程 CPU 和合成数据。真实规模的语料、CUDA 运行、训练显存、跨卡通信、吞吐扩展、开放域模型质量、商业 Jev 等价性仍待各自的独立证据。下一阶段优先固定独立评测集、加入多数类及等预算基线、运行多个种子，再逐步增加样本与模型规模。
