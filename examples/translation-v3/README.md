# 125M 翻译对照实验用句对

[tatoeba.jsonl](tatoeba.jsonl) 含 316 对英语与普通话原句。每行保留 Tatoeba 两侧原文、句子 ID、贡献者及上游署名，采用 **CC BY 2.0 France**。ManyThings 的 2026-02-13 快照 SHA-256 为 `d606361aa7312dcdfb15ef4fc51cb1e7ed8bc2dc608a7b192626cd36e8403106`。本项目筛选、分组并加入翻译指令。作者和原文可通过 `https://tatoeba.org/en/sentences/show/句子ID` 查看。[上游说明](../m2-bilingual-v1/UPSTREAM.md) · [ManyThings](https://www.manythings.org/anki/) · [许可证](https://creativecommons.org/licenses/by/2.0/fr/)

| split | 句对数 | 用途 |
| --- | ---: | --- |
| memory | 12 | 单独权重的双向记忆实验，训练后重答原句 |
| replay | 32 | 各留出实验共用的自然句训练样本 |
| extra-drinks / extra-everyday | 84 / 172 | 仅扩充语料组另用的 256 对自然句 |
| drinks / everyday | 8 / 8 | 训练前人工挑定的近题材与其他日常题，每对只评一个方向 |

受控句由[准备脚本](../../scripts/prepare_translation.py)按中英对照词表生成，包含四个主语、六个动词和六种饮品。Tatoeba 原句与这些组合句在报告中分别计分。[冻结清单](../../docs/experiments/translation-v3-freeze.json)
