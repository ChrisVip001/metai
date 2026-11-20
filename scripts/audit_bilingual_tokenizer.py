"""Audit the exact saved checkpoint tokenizer used for the M2 bilingual run."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer, pre_tokenizers


def audit(checkpoint, data):
    tokenizer = Tokenizer.from_file(str(checkpoint))
    vocab = tokenizer.get_vocab()

    def roundtrip(text):
        # The original ByteLevel pre-tokenizer prepends one ASCII space to these sentences.
        decoded = tokenizer.decode(tokenizer.encode(text).ids)
        return decoded.removeprefix(" ") if not text.startswith(" ") else decoded

    def targets(split):
        for block in (data / f"{split}.txt").read_text().splitlines():
            yield from (
                i for i in tokenizer.encode(block).ids[1:] if i != vocab["<pad>"]
            )

    counts_train = Counter(targets("pretrain"))
    valid = list(targets("valid"))
    total = sum(counts_train.values())
    unigram_nll = sum(
        math.log((total + len(vocab)) / (counts_train[i] + 1)) for i in valid
    ) / len(valid)
    counts = {}
    for line in (data / "attribution.jsonl").read_text().splitlines():
        row = json.loads(line)
        counter = counts.setdefault(
            row["split"], dict(pairs=0, en_changed=0, zh_changed=0)
        )
        counter["pairs"] += 1
        for lang in ("en", "zh"):
            counter[f"{lang}_changed"] += roundtrip(row[lang]) != row[lang]
    missing_digits = [
        digit for digit in "0123456789" if all(digit not in token for token in vocab)
    ]
    altered = []
    impossible = []
    for line in (data / "evaluation.jsonl").read_text().splitlines():
        row = json.loads(line)
        gold = row["output"]
        decoded = roundtrip(gold)
        if decoded != gold:
            altered.append(dict(id=row["id"], reference=gold, roundtrip=decoded))
        if any(d in gold for d in missing_digits):
            impossible.append(row["id"])
    return dict(
        tokenizer_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        vocabulary_size=len(vocab),
        add_one_unigram=dict(
            train_targets=total, valid_targets=len(valid), valid_nll=unigram_nll
        ),
        missing_bytelevel_symbols=sorted(
            set(pre_tokenizers.ByteLevel.alphabet()) - set(vocab)
        ),
        digits_absent_from_all_vocabulary_entries=missing_digits,
        changed_sentences=counts,
        altered_references=altered,
        references_not_representable_verbatim_due_to_absent_ascii_digit=impossible,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tokenizer", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(
        json.dumps(audit(a.tokenizer, a.data), ensure_ascii=False, indent=2) + "\n"
    )
