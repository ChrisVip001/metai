"""Join untouched greedy generations to references, reviewed meanings and training receipts."""

import argparse
import gzip
import html
import itertools
import json
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from prepare_translation import (
    ARCHIVES,
    ARMS,
    FREEZE,
    OBJECTS,
    SUBJECTS,
    SUBSET,
    VERBS,
    sentence,
)
from score_bilingual import normalized, read_jsonl, sha

STAGES = ("baseline", "memory", *ARMS, "mixed")
SLOTS = ("subject", "action", "object", "polarity")
LABELS = ("主语", "动作", "对象", "肯否")
VOCAB = [[x[1] for x in terms] for terms in (SUBJECTS, VERBS, OBJECTS)] + [
    ["肯定", "否定"]
]
REVIEWS = Path("docs/experiments/translation-v3-review.jsonl")


def identity(text):
    return normalized(text).casefold().removesuffix(".").removesuffix("。").strip()


def grammar():
    table = {}
    for values in itertools.product(range(4), range(6), range(6), range(2)):
        pair = sentence(*values)
        for d, lang in (("en-zh", "zh"), ("zh-en", "en")):
            k = (d, identity(pair[lang]))
            if k in table:
                raise ValueError(f"ambiguous reference {k}")
            table[k] = values
    return table


def judge(ref, answer, table, review):
    exact = normalized(ref["output"]) == normalized(answer)
    expected = ref.get("slots")
    observed = table.get((ref["direction"], identity(answer))) if expected else None
    if observed is not None:
        flags = [got == want for got, want in zip(observed, expected)]
        errors = [
            f"{label}应为“{terms[want]}”，译成“{terms[got]}”"
            for label, terms, got, want in zip(LABELS, VOCAB, observed, expected)
            if got != want
        ]
        return dict(
            exact=exact,
            meaning=all(flags),
            slots=dict(zip(SLOTS, flags)),
            reason="。".join(errors) or "主语、动作、对象和肯否均保留。",
        )
    if exact:
        return dict(exact=True, meaning=True, reason="与参考译文一致。")
    entry = review.get((ref["id"], answer))
    if entry is None:
        raise ValueError(f"needs semantic review: {ref['id']} {answer!r}")
    if not isinstance(entry["meaning"], bool) or not entry["reason"].strip():
        raise ValueError(f"incomplete review: {ref['id']}")
    result = dict(exact=False, meaning=entry["meaning"], reason=entry["reason"])
    if expected:
        flags = entry["slots"]
        if set(flags) != set(SLOTS) or any(
            not isinstance(x, bool) for x in flags.values()
        ):
            raise ValueError(f"incomplete slot review: {ref['id']}")
        if entry["meaning"] and not all(flags.values()):
            raise ValueError(f"inconsistent slot review: {ref['id']}")
        result["slots"] = flags
    return result


def contrast(cube, stage, index):
    twins = {}
    slot = SLOTS[index]
    for r in cube:
        s = r["slots"]
        key = (r["direction"], *s[:index], *s[index + 1 :])
        twins.setdefault(key, []).append(r)
    count = complete = 0
    for pair in twins.values():
        if len(pair) != 2 or pair[0]["slots"][index] == pair[1]["slots"][index]:
            raise ValueError(f"unpaired {slot} axis")
        a, b = [r["predictions"][stage] for r in pair]
        # A single generic reply mentioning both alternatives never qualifies.
        count += (
            a["slots"][slot]
            and b["slots"][slot]
            and identity(a["answer"]) != identity(b["answer"])
        )
        complete += a["meaning"] and b["meaning"]
    return dict(slot_both_correct=count, whole_pair_correct=complete, pairs=len(twins))


def corpus_evidence():
    with gzip.open(ARCHIVES[1], "rt") as file:
        prior = [
            row
            for line in file
            if (row := json.loads(line))["split"] in ("sft", "pretrain")
        ]
    verbs = [
        r"lik(?:e[ds]?|ing)",
        r"want(?:s|ed|ing)?",
        r"buy(?:s|ing)?|bought",
        r"drink(?:s|ing)?|drank|drunk",
        r"order(?:s|ed|ing)?",
        r"sell(?:s|ing)?|sold",
    ]
    patterns = [re.compile(r"\b(?:" + expr + r")\b", re.I) for expr in verbs]
    nouns = [re.compile(r"\b" + noun + r"\b", re.I) for noun, _ in OBJECTS]
    prior_sft = [r["en"] for r in prior if r["split"] == "sft"]
    counts = [
        [sum(bool(v.search(s) and o.search(s)) for s in prior_sft) for o in nouns]
        for v in patterns
    ]

    # Inspect lexical proximity for natural test pairs. This is a descriptive audit only.
    # English word matching is not a guarantee against semantic paraphrase in either language.
    corpus = read_jsonl(SUBSET)

    def words(text):
        return re.findall(r"[a-z]+|[0-9]+", text.casefold())

    pool = [
        (r, words(r["en"]))
        for r in prior + corpus
        if r["split"] in ("sft", "pretrain", "replay", "extra-drinks", "extra-everyday")
    ]
    close = []
    for r in corpus:
        if r["split"] not in ("drinks", "everyday"):
            continue
        w = words(r["en"])
        ratio, neighbor = max(
            (
                (SequenceMatcher(None, w, x, autojunk=False).ratio(), other)
                for other, x in pool
            ),
            key=lambda t: (t[0], t[1]["en"]),
        )
        close.append(
            dict(
                en_id=r["en_id"],
                reference_en=r["en"],
                closest_en=neighbor["en"],
                closest_en_id=neighbor["en_id"],
                closest_credit=neighbor["credit"],
                closest_split=neighbor["split"],
                word_sequence_ratio=round(ratio, 3),
            )
        )
    return dict(
        v2_sft_pairs=len(prior_sft),
        verbs=[v for v, _ in VERBS],
        nouns=[o for o, _ in OBJECTS],
        v2_sft_english_cooccurrence=counts,
        closest_natural_in_v2_and_mixed=close,
    )


def report(root, review_path=REVIEWS):
    data = root / "data"
    manifest = json.loads(FREEZE.read_text())
    if json.loads((data / "manifest.json").read_text()) != manifest:
        raise ValueError("benchmark differs from freeze")
    for name, expected in manifest.items():
        if name.endswith(".jsonl") and sha(data / name) != expected:
            raise ValueError(f"changed {name}")
    reviews = read_jsonl(review_path) if review_path.exists() else []
    reviews_by_answer = {(r["id"], r["answer"]): r for r in reviews}
    if len(reviews_by_answer) != len(reviews):
        raise ValueError("duplicated semantic review")
    table = grammar()
    runs = {}
    for name in ("memory", *ARMS, "mixed"):
        run = json.loads((root / f"{name}-run.json").read_text())
        training = json.loads((root / name / "training-report.json").read_text())
        log = (root / f"{name}.log").read_text()
        runs[name] = dict(
            **run,
            **training,
            epoch_mean_batch_loss=[
                float(x) for x in re.findall(r"mean_batch_loss=([0-9.eE+-]+)", log)
            ],
        )
        if (
            run["model_sha256"] != sha(root / name / "model.bpk")
            or run["inputs"]["tokenizer_sha256"] != sha(root / name / "tokenizer.json")
            or training["parameters"] != 125276928
            or training["runtime"] != "Metal"
            or run["inputs"]["data_sha256"] != manifest[f"{name}.jsonl"]
        ):
            raise ValueError(f"unexpected training inputs for {name}")
    if len({r["inputs"]["base_sha256"] for r in runs.values()}) != 1:
        raise ValueError("experiments used different initial weights")
    raw, receipts, scores = {}, {}, {}
    for split, stages in [("memory", ("baseline", "memory")), ("heldout", STAGES)]:
        rows = read_jsonl(data / f"{split}-references.jsonl")
        for name in stages:
            prefix = root / f"{name}-{split}-answers"
            provenance = json.loads(prefix.with_suffix(".provenance.json").read_text())
            expected_model = (
                runs["both"]["inputs"]["base_sha256"]
                if name == "baseline"
                else runs[name]["model_sha256"]
            )
            if (
                provenance["model_sha256"] != expected_model
                or provenance["input_sha256"] != manifest[f"{split}-questions.jsonl"]
                or provenance["binary_sha256"]
                != runs["both"]["inputs"]["binary_sha256"]
            ):
                raise ValueError(f"unverified inference: {name}")
            predicted = read_jsonl(prefix.with_suffix(".jsonl"))
            if [r["id"] for r in rows] != [r["id"] for r in predicted]:
                raise ValueError(f"misaligned {name} / {split}")
            for ref, pred in zip(rows, predicted):
                verdict = judge(ref, pred["answer"], table, reviews_by_answer)
                ref.setdefault("predictions", {})[name] = {
                    **{k: pred[k] for k in ("answer", "token_ids", "stop")},
                    **verdict,
                }
            receipts[f"{name}-{split}"] = {
                **provenance,
                "answers_sha256": sha(prefix.with_suffix(".jsonl")),
            }
            grouped = {}
            for cohort in dict.fromkeys(r["cohort"] for r in rows):
                part = [r for r in rows if r["cohort"] == cohort]
                group = dict(
                    n=len(part),
                    meaning=sum(r["predictions"][name]["meaning"] for r in part),
                    exact=sum(r["predictions"][name]["exact"] for r in part),
                )
                group["meaning_by_direction"] = dict(
                    Counter(
                        r["direction"]
                        for r in part
                        if r["predictions"][name]["meaning"]
                    )
                )
                if cohort == "cube":
                    group["retained"] = {
                        slot: sum(r["predictions"][name]["slots"][slot] for r in part)
                        for slot in SLOTS
                    }
                    group["minimal_pairs"] = {
                        SLOTS[i]: contrast(part, name, i) for i in (1, 2, 3)
                    }
                    group["negative_retained"] = sum(
                        r["predictions"][name]["slots"]["polarity"]
                        for r in part
                        if r["slots"][3]
                    )
                grouped[cohort] = group
            scores.setdefault(name, {}).update(grouped)
        raw[split] = rows
    return dict(
        freeze=manifest,
        corpus_evidence=corpus_evidence(),
        training=runs,
        inference=receipts,
        scores=scores,
        results=raw,
    )


def answers_page(report, dest, report_path):
    report_link = Path(os.path.relpath(report_path, dest.parent)).as_posix()
    method_link = Path(
        os.path.relpath("docs/TRANSLATION.zh-CN.md", dest.parent)
    ).as_posix()

    def cell(text):
        return (
            html.escape(text, quote=False).replace("|", "&#124;").replace("\n", "<br>")
        )

    lines = [
        "# 125M 翻译逐题记录",
        "",
        "同一题的各列均来自该权重的自由生成。✓ 表示整句保留原意，✗ 表示至少一项关键含义出错。"
        f"[所有对照、原始空白、token ID 和理由]({report_link}) · [评测方法]({method_link})",
        "",
    ]
    for split, cohort, heading in [
        ("heldout", "cube", "未见动作与对象的组合"),
        ("heldout", "drinks", "真实饮品句"),
        ("heldout", "everyday", "真实日常句"),
        ("memory", "memory", "重答训练过的 12 个句对"),
    ]:
        lines += [f"## {heading}", ""]
        stages = (
            ["baseline", "memory"]
            if split == "memory"
            else ["baseline", "both", "mixed"]
        )
        labels = {
            "baseline": "上轮 SFT",
            "memory": "记忆训练后",
            "both": "交叉＋肯否成对",
            "mixed": "再扩自然句",
        }
        lines.append(
            "| 题号与原句 | 参考译文 | "
            + " | ".join(labels[k] for k in stages)
            + " | 末列判定依据 |"
        )
        lines.append("|" + " --- |" * (3 + len(stages)))
        credits = []
        for row in report["results"][split]:
            if row["cohort"] != cohort:
                continue
            cells = [row["id"] + "<br>" + cell(row["input"]), cell(row["output"])]
            for stage in stages:
                prediction = row["predictions"][stage]
                cells.append(
                    ("✓ " if prediction["meaning"] else "✗ ")
                    + cell(prediction["answer"])
                )
            cells.append(cell(prediction["reason"]))
            lines.append("| " + " | ".join(cells) + " |")
            if "credit" in row:
                credits.append(row["id"] + " · " + row["credit"])
        if credits:
            lines += ["", "<details><summary>Tatoeba 逐题署名</summary>", ""]
            lines += ["- " + c for c in credits]
            lines += ["", "</details>"]
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("--answers-md", type=Path)
    args = p.parse_args()
    result = report(args.root)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.answers_md:
        args.answers_md.write_text(answers_page(result, args.answers_md, args.report))
