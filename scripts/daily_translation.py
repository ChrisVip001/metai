"""Prepare and run the matched-step natural Chinese–English translation experiment."""

import argparse
from collections import Counter
import gzip
import itertools
import json
import random
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from prepare_bilingual import (
    ARCHIVE_SHA,
    attributed_candidates,
    digest,
    key,
    jsonl,
    task,
)
from prepare_translation import ARCHIVES, DIRECTIONS, SUBSET as V3, sentence
from run_translation import decode, fit
from score_bilingual import read_jsonl, sha

SUBSET = Path("examples/daily-v4/tatoeba.jsonl.gz")
FREEZE = Path("docs/experiments/daily-v4-freeze.json")
ARMS = ("small", "wide", "focused")
RESERVE = Path("examples/daily-v4/reserve.jsonl")
RUBRIC = Path("examples/daily-v4/rubric.jsonl")
PROTOCOL = Path("docs/experiments/daily-v4-protocol.json")
# Each shortlist is ordered before inference. Skip either-side duplicates, including historical variants.
SHORTLIST = {
    "cannot": "1954967 550246 1249224 4666781 2698827 2241048 1954624 17545 414128 3200639",
    "negative": "6439709 321324 1385 3267389 3010330 303895 6531676 2208521 2136815",
    "quantity": "714329 32854 6222208 1492179 8075346 530682 1522032 318031 72904",
    "time": "788259 1488371 4664972 73151 2737521 249413 63290 308975 3819637",
    "other": "11291130 434614 5756645 239100 2649011 1636184 435638 1129494 297626",
}
PATTERNS = {
    "cannot": r"\b(can't|cannot|can not|couldn't|could not|unable|not allowed|not able)\b",
    "negative": r"\b(?:\w+n't|not|never|no|nothing|nobody|neither)\b",
    "quantity": r"\b(?:\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|twice|hundred|half|both|several)\b",
    "time": r"\b(?:yesterday|tomorrow|last|already|now|just|ago|next|still|since|yet|tonight|will)\b|\b(?:am|are|is|was|were|I'm|he's|she's|they're) \w+ing\b",
    "other": r"\?$|^(?:please|bring|give|tell|put|take|turn|open|close|let|show|look|help|call|wait)\b",
}
QUOTAS = dict(cannot=128, negative=320, quantity=256, time=448, other=384)


def english_key(value):
    value = value.casefold().replace("’", "'")
    for a, b in (
        ("won't", "will not"),
        ("can't", "cannot"),
        ("can not", "cannot"),
        ("n't", " not"),
        ("'re", " are"),
        ("'m", " am"),
        ("'ll", " will"),
        ("'ve", " have"),
    ):
        value = value.replace(a, b)
    return key(value)


def identities(rows):
    # ICU is only needed to select/audit the archive. Keep original strings for training and display.
    cmd = shutil.which("uconv") or "/opt/homebrew/opt/icu4c/bin/uconv"
    source = "\n".join(r["zh"] for r in rows) + "\n"
    folded = subprocess.run(
        [cmd, "-x", "Traditional-Simplified"],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    if len(folded) != len(rows):
        raise ValueError("ICU changed line boundaries")
    return [(english_key(r["en"]), key(z)) for r, z in zip(rows, folded)]


def historical():
    rows = read_jsonl(V3)
    for archive in ARCHIVES:
        rows.extend(
            json.loads(line)
            for line in gzip.decompress(archive.read_bytes()).splitlines()
        )
    rows.extend(
        sentence(*args)
        for args in itertools.product(range(4), range(6), range(6), range(2))
    )
    return rows


def select(archive):
    if sha(archive) != ARCHIVE_SHA:
        raise ValueError("unexpected Tatoeba snapshot")
    with zipfile.ZipFile(archive) as z:
        ranked = attributed_candidates(
            z.read("cmn.txt").decode("utf-8-sig").splitlines(), "daily-v4-holdout"
        )
    old = historical()
    seen = set(itertools.chain.from_iterable(identities(old)))
    ids = {r[k] for r in old for k in ("en_id", "zh_id") if k in r}
    rows = [r for _, r in ranked]
    eligible = []
    for r, (en, zh) in zip(rows, identities(rows)):
        if en in seen or zh in seen or {r["en_id"], r["zh_id"]} & ids:
            continue
        ids.update((r["en_id"], r["zh_id"]))
        seen.update((en, zh))
        eligible.append(r)
    by_id = {r["en_id"]: r for r in eligible}
    test = []
    for group, members in SHORTLIST.items():
        part = [dict(by_id[i], group=group) for i in members.split() if i in by_id][:6]
        if len(part) != 6:
            raise ValueError(f"incomplete test group {group}: {len(part)}")
        test.extend(part)
    held = {r["en_id"] for r in test}
    pool = sorted(
        (r for r in eligible if r["en_id"] not in held),
        key=lambda r: digest(("daily-v4-train/" + r["en_id"]).encode()),
    )
    wide = pool[:3072]
    shared = wide[:1536]
    focus = shared.copy()
    used = {r["en_id"] for r in focus}
    for group, n in QUOTAS.items():
        part = [
            r
            for r in pool
            if r["en_id"] not in used and re.search(PATTERNS[group], r["en"], re.I)
        ][:n]
        if len(part) != n:
            raise ValueError(f"not enough eligible examples for {group}: {len(part)}")
        for r in part:
            r["sampling_group"] = group
        used.update(r["en_id"] for r in part)
        focus.extend(part)
    arms = {
        name: {r["en_id"] for r in group}
        for name, group in zip(ARMS, (shared[:384], wide, focus))
    }
    selected = [
        dict(r, arms=[n for n in ARMS if r["en_id"] in arms[n]])
        for r in pool
        if any(r["en_id"] in ids for ids in arms.values())
    ]
    reserves = read_jsonl(RESERVE)
    source = {(r["en_id"], r["zh_id"]): r for r in rows}
    used_rows = old + test + selected
    used_ids = {r[k] for r in used_rows for k in ("en_id", "zh_id") if k in r}
    used_texts = set(itertools.chain.from_iterable(identities(used_rows)))
    for r, keys in zip(reserves, identities(reserves)):
        ids = (r["en_id"], r["zh_id"])
        if {k: v for k, v in r.items() if k != "group"} != source.get(ids):
            raise ValueError("reserve text or credit differs from upstream")
        if used_ids.intersection(ids) or used_texts.intersection(keys):
            raise ValueError("reserve pair overlaps prior data")
    raw = "".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in test + selected
    ).encode()
    SUBSET.parent.mkdir(exist_ok=True, parents=True)
    SUBSET.write_bytes(gzip.compress(raw, mtime=0))


def questions(data, rows, split):
    refs = [
        dict(
            id=f"{r['en_id']}-{d}",
            group=r["group"],
            direction=d,
            credit=r["credit"],
            **task(r, d),
        )
        for r in rows
        for d in DIRECTIONS
    ]
    jsonl(data / f"{split}-references.jsonl", refs)
    jsonl(
        data / f"{split}-questions.jsonl",
        [{k: r[k] for k in ("id", "instruction", "input")} for r in refs],
    )
    return refs


def prepare(root):
    raw = gzip.decompress(SUBSET.read_bytes())
    rows = [json.loads(line) for line in raw.splitlines()]
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    refs = questions(data, (r for r in rows if "group" in r), "heldout")
    reserves = read_jsonl(RESERVE)
    old_ids = {r[k] for r in rows for k in ("en_id", "zh_id")}
    if any(r[k] in old_ids for r in reserves for k in ("en_id", "zh_id")):
        raise ValueError("reserve ID already used")
    extra = questions(data, reserves, "reserve")
    hold = {key(r[k]) for r in refs for k in ("input", "output")}
    if len(refs) != 60 or len({r["id"] for r in refs}) != 60 or len(hold) != 60:
        raise ValueError("invalid bidirectional benchmark")
    hold.update(key(r[k]) for r in extra for k in ("input", "output"))
    if len(hold) != 64:
        raise ValueError("overlapping reserve prompts")
    for name, n in zip(ARMS, (384, 3072, 3072)):
        members = [r for r in rows if name in r.get("arms", [])]
        if len(members) != n or any(
            key(r[k]) in hold for r in members for k in ("en", "zh")
        ):
            raise ValueError(f"invalid split {name}")
        visits = [task(r, d) for r in members for d in DIRECTIONS] * (3072 // n)
        random.Random(42).shuffle(visits)
        jsonl(data / f"{name}.jsonl", visits)
    manifest = dict(
        subset_uncompressed_sha256=digest(raw),
        archive_sha256=ARCHIVE_SHA,
        files={
            f: sha(data / f)
            for f in [
                *(f"{n}.jsonl" for n in ARMS),
                "heldout-references.jsonl",
                "heldout-questions.jsonl",
            ]
        },
        unique_train_pairs={n: sum(n in r.get("arms", []) for r in rows) for n in ARMS},
        common_wide_focused=sum(
            all(n in r.get("arms", []) for n in ("wide", "focused")) for r in rows
        ),
        heldout_pairs=dict(Counter(r["group"] for r in rows if "group" in r)),
        batch=16,
        epochs=4,
        steps=1536,
        seed=42,
        lr=0.00015,
    )
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run(root, base, binary):
    if prepare(root) != json.loads(FREEZE.read_text()):
        raise ValueError("prepared experiment differs from the published freeze")
    protocol = json.loads(PROTOCOL.read_text())
    for file in (
        RESERVE,
        RUBRIC,
        FREEZE,
        root / "data/reserve-questions.jsonl",
        root / "data/reserve-references.jsonl",
    ):
        if sha(file) != protocol["sha256"][file.name]:
            raise ValueError(f"changed {file}")
    for name in ("before", *ARMS):
        model = base
        if name != "before":
            model = fit(root, name, base, binary, schedule=(16, 4))
            stats = json.loads((model / "training-report.json").read_text())
            if (stats["runtime"], stats["parameters"], stats["optimizer_steps"]) != (
                "Metal",
                125276928,
                1536,
            ):
                raise ValueError(f"training receipt mismatch for {name}")
        for split in ("heldout", "reserve"):
            decode(root, binary, name, model, split)
        print(f"finished {name}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--select-from", type=Path)
    p.add_argument("--train", action="store_true")
    p.add_argument("--base", type=Path, default=Path("artifacts/translation-v3/mixed"))
    p.add_argument(
        "--binary", type=Path, default=Path("artifacts/m2-bilingual-v2/metai-metal")
    )
    a = p.parse_args()
    if a.select_from:
        select(a.select_from)
    if a.train:
        run(a.root, a.base, a.binary)
    else:
        print(json.dumps(prepare(a.root), indent=2))
