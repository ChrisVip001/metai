"""Verify a saved tokenizer on the byte alphabet, Unicode scalars, mixed strings and real data."""

import argparse
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer, pre_tokenizers


def verify(path, corpus):
    t = Tokenizer.from_file(str(path))
    alphabet = set(pre_tokenizers.ByteLevel.alphabet())
    missing = sorted(alphabet - set(t.get_vocab()))
    if missing:
        raise ValueError(f"missing {len(missing)} byte symbols")

    def check(s):
        ids = t.encode(s, add_special_tokens=False).ids
        if t.decode(ids, skip_special_tokens=False) != s:
            raise ValueError(f"round-trip failure in {s[:24]!r}")

    mixed = [
        "",
        "中文 English 0123456789",
        "青橙47",
        "MAPLE-731",
        "0#1%2+3/4@5<6=7>8\\9}",
        "  A\t中\r\n B  ",
        "e\u0301 / é",
        "𠮷🙂🚀",
        "<|user|>\n<pad><s></s><unk>\n<|assistant|>",
        "".join(map(chr, range(128))),
    ]
    for s in mixed:
        check(s)
    scalars = 0
    for start, end in [(0, 0xD800), (0xE000, 0x110000)]:
        for i in range(start, end, 256):
            text = "".join(map(chr, range(i, min(i + 256, end))))
            check(text)
            scalars += len(text)
    sentences = 0
    for line in corpus.read_text().splitlines():
        row = json.loads(line)
        for lang in ("en", "zh"):
            check(row[lang])
            sentences += 1
    questions = [
        json.loads(line)
        for line in corpus.with_name("evaluation.jsonl").read_text().splitlines()
    ]
    for row in questions:
        for field in ("instruction", "input", "output"):
            check(row[field])
    return dict(
        tested_question_triplets=len(questions),
        tokenizer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        vocabulary=t.get_vocab_size(),
        byte_symbols=len(alphabet),
        unicode_scalars=scalars,
        mixed_cases=mixed,
        real_sentences=sentences,
        mismatches=0,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("tokenizer", "attribution", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    report = verify(a.tokenizer, a.attribution)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report)
