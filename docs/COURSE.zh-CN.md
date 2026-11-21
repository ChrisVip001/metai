# MetaI 中文教程

## 1. 训练自己的模型

先记录语料来源、授权和版本。预训练用每行一段的 UTF-8 文本，或含 `text` 字段的 JSONL。隔离训练、验证和测试，并对两侧原文做规范化与近重复排查。SFT 的 JSONL 每行包含 `instruction`、`output`，可选 `input`。测试题只提供指令与原句。

以下示例使用 Metal。CUDA 环境将构建特性与设备名换成 `cuda`。把 `DATA` 改成自己的数据路径，并为每次训练选择新目录。

```bash
DATA=/path/to/my-data
RUN=artifacts/my-run
CAP=64
cargo build --locked --no-default-features --features metal --bin metai
mkdir -p "$RUN"
B=target/debug/metai
python3 scripts/run_size_tiers.py config "$CAP" > "$RUN/config.json"
"$B" --device metal pretrain --config "$RUN/config.json" --tokenizer-path tokenizer.json \
  --train-path "$DATA/train.txt" --valid-path "$DATA/valid.txt" \
  --batch-size 16 --num-epochs 1 --output-dir "$RUN/base"
python3 scripts/run_size_tiers.py check "$CAP" "$RUN/base"
"$B" --device metal train-sft --model-dir "$RUN/base" --data-path "$DATA/sft.jsonl" \
  --batch-size 16 --num-epochs 4 --learning-rate 0.00015 --output-dir "$RUN/sft"
python3 scripts/run_size_tiers.py check "$CAP" "$RUN/sft"
"$B" --device metal eval-chat --model-dir "$RUN/sft" \
  --questions "$DATA/questions.jsonl" --max-len 32
```

`questions.jsonl` 每行提供 `id`、`instruction` 和 `input`。输出包含 `answer`、生成 token ID 和停止原因。自由生成的中英双向译文应分别核对整句原意、动作、对象、否定和数量。先验证分词往返，再固定参考译文、置换源句，对照 NLL 与实际生成结果。[置换评分工具](../examples/score_source.rs)

## 2. 读懂训练链路

| 阶段 | 核心检查 | 入口 |
| --- | --- | --- |
| 预训练 | token 序列错位一位形成目标，padding 从损失中排除，验证 NLL 按目标 token 归约 | `pretrain`、`resume-pretrain` |
| SFT 与 LoRA | 只监督回答和结束符，LoRA 用低秩路径更新 | `train-sft`、`train-lora` |
| 偏好与强化学习 | 偏好对共享问题，在线训练分开核对采样、奖励、优势及策略更新 | `train-dpo`、`train-ppo`、`train-grpo`、`train-gspo`、`train-cispo` |
| 自适应推理与蒸馏 | 分别记录答案质量、推理长度和资源 | `train-thinking`、`think`、`distill` |
| 工具、决策与模态 | 验证工具轨迹、校准及各模态消融 | `train-agent`、`train-decision`、`train-multimodal`、`lab` |

输出目录保存 `config.json`、`tokenizer.json` 和 `model.bpk`。完整预训练恢复还需要对应的优化器和进度。纯权重文件大小不含训练梯度、优化器和 KV 缓存。

## 3. 运行自动生成的回归示例

课程按需在临时目录创建算术、颜色和视听输入，用于复查训练、保存与加载。决策和视听对照另行生成留出输入。

```bash
WORK=$(mktemp -d)
python3 scripts/run_tiny.py --output "$WORK/course"
python3 scripts/compare_tiny.py --course "$WORK/course" --output "$WORK/holdout"
```

决策对照检查同一名词换颜色时，标签与置信度是否随输入变化。视听对照固定一侧输入、只换另一侧，核对回答、结束符及无效字节。使用 `cargo run --locked -- <命令> --help` 查询参数。

[返回首页](../README.md) · [工程规格](INDUSTRIAL.zh-CN.md)
