# MetaI 双语 125M 实验数据及署名

[subset.jsonl.gz](subset.jsonl.gz) 保留这次实验用到的 2,352 个英语和普通话句对。原作者是 Tatoeba 社区的贡献者，由 ManyThings 整理。句子按 **CC BY 2.0 France** 提供。语料的许可独立于框架代码的 MIT 许可。[原作者平台及使用条款](https://tatoeba.org/en/terms_of_use) · [ManyThings 双语数据](https://www.manythings.org/anki/) · [CC BY 2.0 France](https://creativecommons.org/licenses/by/2.0/fr/) · [原数据包说明](UPSTREAM.md)

解压后每一行列出原文、两位贡献者的用户名、两个句子 ID、原始署名和实验划分。访问 `https://tatoeba.org/en/sentences/show/{ID}` 可追溯单句。在派生语料中保留这些字段。MetaI 对句子做了长度过滤和去重，用句界符拼成训练段落，另给部分句对添加翻译和语种识别指令。准备脚本可逐字节恢复本轮所用数据，包括问题和参考答案。

原压缩包标注日期为 2026-02-13。原包 SHA-256 为 `d606361aa7312dcdfb15ef4fc51cb1e7ed8bc2dc608a7b192626cd36e8403106`。子集解压后的 SHA-256 为 `eb1d8e34317d8a236b7bd976f17a93f44c8a87c38025da74359c524214c3f2b8`。

历史 125M 权重使用的 16K 词表也保存在 [tokenizer.json.gz](tokenizer.json.gz)，解压后 SHA-256 为 `3acf8aeefcfa48cb3bff94529566dfdfe89dd67fe9d19c3d25f19c9fb197e51f`。重放 v1 的预训练时请按教程显式传入此文件。
