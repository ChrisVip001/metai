# 真实日常翻译语料与授权

训练与留出句对均来自 ManyThings 2026-02-13 发布的普通话／英语 Tatoeba 快照。只选带有 CC BY 2.0 France 声明、双方作者名和句子 ID 的行。选取时保留原文，训练时加入翻译指令并交换源语言和目标语言。两侧贡献者保存在每条记录的 `en_author`、`zh_author` 和上游 `credit` 中。[ManyThings 发布页](https://www.manythings.org/anki/) · [CC BY 2.0 France](https://creativecommons.org/licenses/by/2.0/fr/) · [快照自带的说明](../m2-bilingual-v1/UPSTREAM.md)

| 文件 | 内容 |
| --- | --- |
| [tatoeba.jsonl.gz](tatoeba.jsonl.gz) | 3,890 个不同训练句对及最初冻结的 30 个测试句对。训练行的 `arms` 记录哪些组使用了该句对，测试行有 `group` 标签 |
| [reserve.jsonl](reserve.jsonl) | 两个后备测试句对，与主数据采用相同的署名格式 |
| [rubric.jsonl](rubric.jsonl) | 每对测试句考查的关键含义。整句是否正确另行判定 |

训练组共有 384、3,072、3,072 个不同句对。后两组共享 2,254 对。解压后内容 SHA-256 为 `df7f01e863da2cd016a4bcb84b1509358a141eac310353b2448be9f797e0a218`。按 ID 可在 `https://tatoeba.org/en/sentences/show/句子ID` 查看原句。[训练清单](../../docs/experiments/daily-v4-freeze.json) · [主评测与近似句的划分](../../docs/experiments/daily-v4-protocol.json)
