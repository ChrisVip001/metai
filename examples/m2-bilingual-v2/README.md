# MetaI 双语实验 v2 语料及署名

[subset.jsonl.gz](subset.jsonl.gz) 存放 11,456 个英语／普通话原句对。Tatoeba 社区贡献者拥有各句署名，ManyThings 整理了 2026-02-13 版双语快照。原句按 CC BY 2.0 France 提供，解压后每行附 `en_author`、`zh_author`、两个句子 ID 和上游署名。可将 ID 附加到 `https://tatoeba.org/en/sentences/show/` 查询原句。[Tatoeba 条款](https://tatoeba.org/en/terms_of_use) · [ManyThings 来源](https://www.manythings.org/anki/) · [上游说明](../m2-bilingual-v1/UPSTREAM.md) · [CC BY 2.0 France](https://creativecommons.org/licenses/by/2.0/fr/)

MetaI 对原句进行了筛选、去重、分组，并保留原文。预训练将对应的中英句子拼接，SFT 为每对生成双向翻译指令，另取一部分生成语种识别指令。测试题与诊断题在选取训练语料前划出。子集解压后的 SHA-256 为 `6c6c4fa61dc3eae5676d42991121ea8aadfd73cbd9f33d0ddf41bff9414f4e78`。[准备脚本](../../scripts/prepare_bilingual_v2.py)
