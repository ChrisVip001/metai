"""Freeze the larger Tatoeba corpus and a benchmark disjoint from the entire v1 subset."""

import argparse
import gzip
import json
import zipfile
from pathlib import Path

from prepare_bilingual import (
    ARCHIVE_SHA,
    LICENSE,
    SOURCE,
    SUBSET_SHA,
    attributed_candidates,
    digest,
    jsonl,
    key,
    task,
)

SIZES = {"test": 32, "dev": 32, "valid": 128, "sft": 3072, "pretrain": 8192}
TRANSFER = [
    ("arithmetic", "三加一等于多少？只写数字。", "4"),
    ("arithmetic", "What is eight minus seven? Answer with one digit.", "1"),
    ("knowledge", "一年有几个月？只写阿拉伯数字。", "12"),
    (
        "knowledge",
        "What is the capital of Italy? Give only the English city name.",
        "Rome",
    ),
    ("reading", "红箱装书，白箱装鞋。鞋在哪个箱里？只写颜色。", "白"),
    (
        "reading",
        "The scarf is in the bag. The hat is on the desk. Where is the hat? Answer with one noun.",
        "desk",
    ),
    ("copy", "原样抄写，不要添加文字：雨伞90#", "雨伞90#"),
    ("copy", "Copy exactly, with no extra text: KITE@418", "KITE@418"),
]
V1 = Path(__file__).resolve().parents[1] / "examples/m2-bilingual-v1/subset.jsonl.gz"


def select(archive):
    if digest(archive.read_bytes()) != ARCHIVE_SHA:
        raise ValueError("unexpected upstream archive digest")
    legacy = gzip.decompress(V1.read_bytes())
    if digest(legacy) != SUBSET_SHA:
        raise ValueError("unexpected v1 subset digest")
    seen = {key(prompt) for _, prompt, _ in TRANSFER}
    ids = set()
    for line in legacy.splitlines():
        row = json.loads(line)
        for lang in ("en", "zh"):
            seen.add(key(row[lang]))
            ids.add(row[lang + "_id"])
    with zipfile.ZipFile(archive) as z:
        pool = attributed_candidates(
            z.read("cmn.txt").decode("utf-8-sig").splitlines(), "metai-bilingual-v2"
        )
    splits = {}
    candidates = iter(pool)
    for split, n in SIZES.items():
        rows = []
        while len(rows) < n:
            _, row = next(candidates)
            keys = {key(row["en"]), key(row["zh"])}
            new_ids = {row["en_id"], row["zh_id"]}
            if keys & seen or new_ids & ids:
                continue
            rows.append(row)
            seen.update(keys)
            ids.update(new_ids)
        splits[split] = rows
    return splits


def prepare(source, out):
    out.mkdir(parents=True, exist_ok=False)
    if source.suffix == ".gz":
        splits = {s: [] for s in SIZES}
        raw = gzip.decompress(source.read_bytes())
        if (
            digest(raw)
            != "6c6c4fa61dc3eae5676d42991121ea8aadfd73cbd9f33d0ddf41bff9414f4e78"
        ):
            raise ValueError("unexpected frozen v2 subset digest")
        for line in raw.splitlines():
            row = json.loads(line)
            splits[row.pop("split")].append(row)
        if {s: len(v) for s, v in splits.items()} != SIZES:
            raise ValueError("incorrect frozen split sizes")
    else:
        splits = select(source)
    jsonl(
        out / "attribution.jsonl",
        [dict(split=s, **r) for s, rows in splits.items() for r in rows],
    )
    for split in ("pretrain", "valid"):
        rows = splits[split]
        blocks = []
        for i in range(0, len(rows), 2):
            r, t = rows[i : i + 2]
            blocks.append(" </s> ".join([r["en"], r["zh"], t["zh"], t["en"]]) + " </s>")
        (out / f"{split}.txt").write_text("\n".join(blocks) + "\n", encoding="utf8")
    train = []
    for i, r in enumerate(splits["sft"]):
        train.extend(task(r, k) for k in ("en-zh", "zh-en"))
        if i % 8 == 0:
            train.extend(task(r, k) for k in ("lang-en", "lang-zh"))
    jsonl(out / "sft.jsonl", train)
    evaluation = []
    for i, r in enumerate(splits["test"]):
        k = ("en-zh", "zh-en")[i % 2]
        evaluation.append(
            dict(id=f"{k}-{i:02}", kind=k, **task(r, k), credit=r["credit"])
        )
        if i < 16:
            k = ("lang-en", "lang-zh")[i % 2]
            evaluation.append(
                dict(id=f"{k}-{i:02}", kind=k, **task(r, k), credit=r["credit"])
            )
    evaluation.extend(
        dict(id=f"transfer-{i:02}", kind=k, instruction=q, input="", output=a)
        for i, (k, q, a) in enumerate(TRANSFER)
    )
    jsonl(out / "evaluation.jsonl", evaluation)
    jsonl(
        out / "questions.jsonl",
        [{k: r[k] for k in ("id", "instruction", "input")} for r in evaluation],
    )
    # Teacher-forced contrast on separate development pairs; identical target in each comparison.
    diagnostic = []
    for i, r in enumerate(splits["dev"]):
        kind = ("en-zh", "zh-en")[i % 2]
        original = task(r, kind)
        wrong = task(splits["dev"][(i + 2) % len(splits["dev"])], kind)["input"]
        diagnostic.append(
            dict(id=f"dev-{i:02}", kind=kind, **original, swapped_input=wrong)
        )
    jsonl(out / "contrast.jsonl", diagnostic)
    manifest = dict(
        source=SOURCE,
        archive_sha256=ARCHIVE_SHA,
        license=LICENSE,
        release="2026-02-13",
        split_pairs=SIZES,
        files={p.name: digest(p.read_bytes()) for p in sorted(out.iterdir())},
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    print(prepare(a.source, a.out))
