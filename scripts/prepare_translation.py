"""Freeze the translation factorial, verbatim Tatoeba subset and source-only prompts."""

import argparse
import gzip
import itertools
import json
import random
import re
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

FREEZE = Path("docs/experiments/translation-v3-freeze.json")
SUBSET = Path("examples/translation-v3/tatoeba.jsonl")
ARCHIVES = [
    Path(f"examples/m2-bilingual-v{version}/subset.jsonl.gz") for version in (1, 2)
]
SUBJECTS = [("I", "我"), ("Tom", "汤姆"), ("Nina", "妮娜"), ("Leo", "利奥")]
VERBS = [
    ("like", "喜欢"),
    ("want", "想要"),
    ("buy", "买"),
    ("drink", "喝"),
    ("order", "点"),
    ("sell", "卖"),
]
OBJECTS = [
    ("coffee", "咖啡"),
    ("tea", "茶"),
    ("milk", "牛奶"),
    ("juice", "果汁"),
    ("water", "水"),
    ("beer", "啤酒"),
]
DIRECTIONS = ("en-zh", "zh-en")
ARMS = ("fixed", "crossed", "polarity", "both")
DRINK = re.compile(r"\b(coffee|tea|milk|juice|water|beer)\b", re.I)
RELATED = re.compile(
    r"\b(like|want|buy|bought|drink|order|sell|sold|doesn't|don't|can't|not|never)\b",
    re.I,
)
# Short, attributed sentences selected before inference. Alternate translation direction within each stratum.
REAL_IDS = {
    "drinks": "3132269 1761849 1084522 252024 2239680 253436 62229 1615236".split(),
    "everyday": "1315801 3301001 25765 8812308 63744 1744777 4664279 312724".split(),
}


def sentence(s, a, o, neg):
    who, subject = SUBJECTS[s]
    verb, action = VERBS[a]
    noun, obj = OBJECTS[o]
    if neg:
        predicate = ("do not " if s == 0 else "does not ") + verb
    else:
        predicate = verb if s == 0 else verb + "s"
    return dict(
        en=f"{who} {predicate} {noun}.",
        zh=subject + ("不" if neg else "") + action + obj + "。",
    )


def previous():
    seen, ids = set(), set()
    for archive in ARCHIVES:
        with gzip.open(archive, "rt", encoding="utf-8") as fp:
            for line in fp:
                r = json.loads(line)
                seen.update(key(r[k]) for k in ("en", "zh"))
                ids.update(r[k] for k in ("en_id", "zh_id"))
    return seen, ids


def texts(row):
    return {key(row[k]) for k in ("en", "zh")}


def cubes():
    for s, actions, objects in [(2, (0, 1), (3, 4)), (3, (3, 4), (0, 1))]:
        yield from itertools.product([s], actions, objects, (0, 1))


def select(archive):
    if digest(archive.read_bytes()) != ARCHIVE_SHA:
        raise ValueError("unexpected upstream archive")
    with zipfile.ZipFile(archive) as z:
        candidates = [
            r
            for _, r in attributed_candidates(
                z.read("cmn.txt").decode("utf-8-sig").splitlines(),
                "metai-translation-v3",
            )
        ]
    seen, ids = previous()
    # Reserve even unused formula sentences, so the human corpus cannot duplicate either side.
    for args in itertools.product(range(4), range(6), range(6), range(2)):
        seen.update(texts(sentence(*args)))
    chosen = []

    def take(label, n, predicate):
        count = 0
        for r in candidates:
            pair_ids = {r["en_id"], r["zh_id"]}
            if (
                len(r["en"].split()) > 8
                or pair_ids & ids
                or texts(r) & seen
                or not predicate(r)
            ):
                continue
            chosen.append(dict(split=label, **r))
            seen.update(texts(r))
            ids.update(pair_ids)
            count += 1
            if count == n:
                return
        raise ValueError(f"could only find {count} examples for {label}")

    for label, members in REAL_IDS.items():
        for en_id in members:
            take(label, 1, lambda r: r["en_id"] == en_id)
    take(
        "memory",
        12,
        lambda r: 3 <= len(r["en"].split()) <= 6 and not DRINK.search(r["en"]),
    )
    take("replay", 32, lambda r: not DRINK.search(r["en"]))
    take("extra-drinks", 84, lambda r: DRINK.search(r["en"]))
    take(
        "extra-everyday",
        172,
        lambda r: RELATED.search(r["en"]) and not DRINK.search(r["en"]),
    )
    SUBSET.parent.mkdir(exist_ok=True, parents=True)
    jsonl(SUBSET, chosen)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    corpus = [json.loads(line) for line in SUBSET.read_text().splitlines()]
    prior, ids = previous()
    natural = set()
    for row in corpus:
        pair_ids = {row["en_id"], row["zh_id"]}
        if pair_ids & ids or texts(row) & (prior | natural):
            raise ValueError(f"reused Tatoeba row {row['en_id']}")
        ids.update(pair_ids)
        natural.update(texts(row))
    evaluation, memory = [], []

    def question(rows, pair, cohort, direction, **details):
        rows.append(
            dict(
                id=f"{cohort}-{len(rows)+1:02d}",
                cohort=cohort,
                direction=direction,
                **task(pair, direction),
                **details,
            )
        )

    for s, a, o, n in cubes():
        pair = sentence(s, a, o, n)
        if texts(pair) & (prior | natural):
            raise ValueError(f"held-out sentence already used: {pair}")
        for direction in DIRECTIONS:
            question(evaluation, pair, "cube", direction, slots=[s, a, o, n])
    for label in REAL_IDS:
        for i, row in enumerate(r for r in corpus if r["split"] == label):
            question(evaluation, row, label, DIRECTIONS[i % 2], credit=row["credit"])
    for row in (r for r in corpus if r["split"] == "memory"):
        for direction in DIRECTIONS:
            question(memory, row, "memory", direction, credit=row["credit"])
    for name, rows in (("heldout", evaluation), ("memory", memory)):
        jsonl(out / f"{name}-references.jsonl", rows)
        jsonl(
            out / f"{name}-questions.jsonl",
            [{k: r[k] for k in ("id", "instruction", "input")} for r in rows],
        )
    jsonl(
        out / "memory.jsonl",
        [{k: r[k] for k in ("instruction", "input", "output")} for r in memory],
    )
    heldout_texts = {key(r[k]) for r in evaluation for k in ("input", "output")}
    replay = [task(r, d) for r in corpus if r["split"] == "replay" for d in DIRECTIONS]
    for arm in (*ARMS, "mixed"):
        rows = []
        for s, a, delta, sign in itertools.product(
            range(4), range(6), (-1, 0, 1), (0, 1)
        ):
            o = (a + delta) % 6 if arm in ("crossed", "both", "mixed") else a
            neg = sign if arm in ("polarity", "both", "mixed") else (s + a) % 2
            for d in DIRECTIONS:
                rows.append(task(sentence(s, a, o, neg), d))
        rows += replay
        if arm == "mixed":
            rows += [
                task(r, d)
                for r in corpus
                if r["split"].startswith("extra-")
                for d in DIRECTIONS
            ]
        if any(key(r[k]) in heldout_texts for r in rows for k in ("input", "output")):
            raise ValueError(f"leakage in {arm}")
        random.Random(42).shuffle(rows)
        jsonl(out / f"{arm}.jsonl", rows)
    names = [
        f"{s}.jsonl"
        for s in (
            *ARMS,
            "mixed",
            "memory",
            "memory-questions",
            "memory-references",
            "heldout-questions",
            "heldout-references",
        )
    ]
    manifest = {f: digest((out / f).read_bytes()) for f in names}
    manifest["tatoeba_subset"] = digest(SUBSET.read_bytes())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--select-from",
        type=Path,
        help="regenerate the frozen attributed subset from the pinned ZIP",
    )
    args = p.parse_args()
    if args.select_from:
        select(args.select_from)
    prepare(args.out)
