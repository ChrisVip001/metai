"""Prepare the pinned, attributed Tatoeba snapshot for the M2 Pro 125M run."""

import argparse
import gzip
import hashlib
import json
import re
import unicodedata
import zipfile
from pathlib import Path

ARCHIVE_SHA = "d606361aa7312dcdfb15ef4fc51cb1e7ed8bc2dc608a7b192626cd36e8403106"
SUBSET_SHA = "eb1d8e34317d8a236b7bd976f17a93f44c8a87c38025da74359c524214c3f2b8"
LICENSE = "https://creativecommons.org/licenses/by/2.0/fr/"
SOURCE = "https://www.manythings.org/anki/cmn-eng.zip"
CREDIT = re.compile(
    r"CC-BY 2.0 \(France\) Attribution: tatoeba.org #(\d+) \(([^()]+)\) & #(\d+) \(([^()]+)\)"
)
TRANSFER = [
    ("arithmetic", "七加五等于多少？只写数字。", "12"),
    ("arithmetic", "What is nine minus four? Give only the digit.", "5"),
    ("knowledge", "水在标准大气压下多少摄氏度结冰？只写数字。", "0"),
    ("knowledge", "What is the capital of Japan? Give only the English name.", "Tokyo"),
    (
        "reading",
        "小林把钥匙放在蓝盒里，把笔放在红盒里。钥匙在哪个盒里？只答颜色。",
        "蓝",
    ),
    (
        "reading",
        "The mug is on the shelf. The spoon is in the drawer. Where is the spoon? Answer with one noun.",
        "drawer",
    ),
    ("copy", "原样抄写这串代号，不要添加文字：青橙47", "青橙47"),
    ("copy", "Copy this code exactly and add nothing else: MAPLE-731", "MAPLE-731"),
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def key(text):
    return "".join(
        c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum()
    )


def jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def task(row, kind):
    en, zh = row["en"], row["zh"]
    instruction, source, target = {
        "en-zh": ("把这句话译成中文。", en, zh),
        "zh-en": ("Translate this sentence into English.", zh, en),
        "lang-en": ("判断这句话的语种，只回答中文或英文。", en, "英文"),
        "lang-zh": ("判断这句话的语种，只回答中文或英文。", zh, "中文"),
    }[kind]
    return dict(instruction=instruction, input=source, output=target)


def attributed_candidates(lines, salt="metai-bilingual-v1"):
    candidates = []
    for line in lines:
        en, zh, credit = line.split("\t")
        match = CREDIT.fullmatch(credit)
        if not match or not (3 <= len(en.split()) <= 14 and 3 <= len(zh) <= 40):
            continue
        if not (
            en.isascii()
            and re.search(r"[A-Za-z]", en)
            and re.search(r"[\u4e00-\u9fff]", zh)
        ):
            continue
        if any(x.lower() in {"", "unknown", "\\n"} for x in (match[2], match[4])):
            continue
        rank = digest((salt + "/" + match[1] + "/" + match[3]).encode())
        candidates.append(
            (
                rank,
                dict(
                    en=en,
                    zh=zh,
                    en_id=match[1],
                    en_author=match[2],
                    zh_id=match[3],
                    zh_author=match[4],
                    credit=credit,
                ),
            )
        )
    return sorted(candidates, key=lambda item: item[0])


def prepare(archive, out, pairs, sft_pairs):
    if digest(archive.read_bytes()) != ARCHIVE_SHA:
        raise ValueError(
            "Archive differs from the February 13, 2026 snapshot; check release and attribution"
        )
    if pairs < 32 or pairs % 4 or sft_pairs < 4 or sft_pairs % 4:
        raise ValueError("Select multiples of four and at least 32 pretraining pairs")
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        lines = z.read("cmn.txt").decode("utf-8-sig").splitlines()
        (out / "UPSTREAM.txt").write_bytes(z.read("_about.txt"))
    seen = {key(q) for _, q, _ in TRANSFER}
    chosen = []
    for _, row in attributed_candidates(lines):
        keys = {key(row["en"]), key(row["zh"])}
        if keys & seen:
            continue
        seen.update(keys)
        chosen.append(row)
    # Assign evaluation first, so changing the training budget cannot move evaluation rows.
    plan = [("test", 16), ("valid", 32), ("sft", sft_pairs), ("pretrain", pairs)]
    if len(chosen) < sum(n for _, n in plan):
        raise ValueError("not enough unique attributed pairs")
    splits = {}
    offset = 0
    for name, n in plan:
        splits[name] = chosen[offset : offset + n]
        offset += n
    return write_splits(out, splits)


def from_subset(archive, out):
    # The committed subset keeps the exact benchmark reproducible if the rolling URL changes.
    raw = gzip.decompress(archive.read_bytes())
    if digest(raw) != SUBSET_SHA:
        raise ValueError("unexpected subset digest")
    splits = {name: [] for name in ("test", "valid", "sft", "pretrain")}
    for line in raw.splitlines():
        row = json.loads(line)
        splits[row.pop("split")].append(row)
    out.mkdir(parents=True, exist_ok=False)
    (out / "UPSTREAM.txt").write_bytes(archive.with_name("UPSTREAM.md").read_bytes())
    return write_splits(out, splits)


def write_splits(out, splits):
    kinds = ["en-zh", "zh-en", "lang-en", "lang-zh"]
    jsonl(
        out / "sft.jsonl", [task(r, kinds[i % 4]) for i, r in enumerate(splits["sft"])]
    )
    evaluation = []
    for i, r in enumerate(splits["test"]):
        for kind in (kinds[i % 2], kinds[2 + i % 2]):
            evaluation.append(
                dict(
                    id=f"{kind}-{i+1:02}",
                    kind=kind,
                    **task(r, kind),
                    credit=r["credit"],
                )
            )
    for i, (kind, question, answer) in enumerate(TRANSFER):
        evaluation.append(
            dict(
                id=f"transfer-{i+1:02}",
                kind=kind,
                instruction=question,
                input="",
                output=answer,
            )
        )
    jsonl(out / "evaluation.jsonl", evaluation)
    jsonl(
        out / "questions.jsonl",
        [{k: r[k] for k in ("id", "instruction", "input")} for r in evaluation],
    )
    for split in ("pretrain", "valid"):
        data = splits[split]
        # Pack four complete sentences in a single language per line, with explicit document boundaries.
        blocks = [
            " </s> ".join(r[lang] for r in data[i : i + 4]) + " </s>"
            for i in range(0, len(data), 4)
            for lang in ("en", "zh")
        ]
        (out / f"{split}.txt").write_text("\n".join(blocks) + "\n", encoding="utf-8")
        if split == "pretrain":
            (out / "pilot.txt").write_text(
                "\n".join(blocks[:16]) + "\n", encoding="utf-8"
            )
    jsonl(
        out / "attribution.jsonl",
        [dict(split=name, **row) for name, rows in splits.items() for row in rows],
    )
    manifest = dict(
        source=SOURCE,
        release="2026-02-13",
        archive_sha256=ARCHIVE_SHA,
        license=LICENSE,
        upstream_rows=32028,
        split_pairs={name: len(rows) for name, rows in splits.items()},
        files={p.name: digest(p.read_bytes()) for p in sorted(out.iterdir())},
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--pretrain-pairs", type=int, default=2048)
    p.add_argument("--sft-pairs", type=int, default=256)
    args = p.parse_args()
    if args.archive.suffix == ".gz":
        if (args.pretrain_pairs, args.sft_pairs) != (2048, 256):
            p.error("the frozen subset has a fixed training budget")
        manifest = from_subset(args.archive, args.out)
    else:
        manifest = prepare(args.archive, args.out, args.pretrain_pairs, args.sft_pairs)
    print(manifest["split_pairs"])
