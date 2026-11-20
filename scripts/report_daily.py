"""Blind review and verified per-direction reporting for the real daily translation run."""

import argparse
from collections import Counter
import gzip
import html
import json
import os
import re
from pathlib import Path

from daily_translation import (
    ARMS,
    FREEZE,
    PATTERNS,
    PROTOCOL,
    RUBRIC,
    SUBSET,
)
from prepare_bilingual import digest, jsonl
from score_bilingual import normalized, read_jsonl, sha

STAGES = ("before", *ARMS)
GROUPS = dict(
    cannot="不能／做不到",
    negative="一般否定",
    quantity="数量",
    time="时间与体",
    other="其他日常表达",
)
REVIEWS = Path("docs/experiments/daily-v4-review.jsonl")
TOPICS = {
    "家庭": r"\b(?:family|parents?|father|mother|children|son|daughter|wife|husband|sister|brother)\b",
    "学校": r"\b(?:school|class|teacher|students?|study|studying|homework|college|university|library|test|exam)\b",
    "工作": r"\b(?:work|job|company|meeting|office|boss|business|working)\b",
    "交通与出行": r"\b(?:car|bus|train|plane|bike|bicycle|street|road|travel|trip|station|ticket|airport)\b",
    "饮食": r"\b(?:food|eat|ate|eating|drink|dinner|breakfast|lunch|rice|meat|fruit|coffee|milk|tea|water)\b",
    "购物与钱": r"\b(?:buy|bought|sell|sold|shop|store|money|dollars?|price|cost|pay|paid|gift)\b",
    "健康": r"\b(?:sick|ill|hospital|doctor|sleep|slept|pain|medicine|health|tired)\b",
    "休闲": r"\b(?:sport|game|play|played|music|book|film|movie|swim|tennis|television)\b",
}


def inputs(root):
    data = root / "data"
    manifest = json.loads(FREEZE.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    diagnostic_ids = set(protocol["diagnostic_ids"])
    for key, h in manifest["files"].items():
        if sha(data / key) != h:
            raise ValueError(f"changed experiment file {key}")
    for filename, h in protocol["sha256"].items():
        file = {
            "daily-v4-freeze.json": FREEZE,
            "rubric.jsonl": RUBRIC,
            "reserve.jsonl": RUBRIC.with_name("reserve.jsonl"),
        }.get(filename, data / filename)
        if sha(file) != h:
            raise ValueError(f"changed protocol file {file}")
    if (
        digest(gzip.decompress(SUBSET.read_bytes()))
        != manifest["subset_uncompressed_sha256"]
    ):
        raise ValueError("changed corpus")
    rubric = {r["en_id"]: r["detail"] for r in read_jsonl(RUBRIC)}
    rows = []
    receipts = {}
    for split in ("heldout", "reserve"):
        refs = read_jsonl(data / f"{split}-references.jsonl")
        for r in refs:
            pair = r["id"].split("-")[0]
            r.update(
                pair=pair,
                primary=pair not in diagnostic_ids,
                criterion=rubric[pair],
                predictions={},
            )
        for stage in STAGES:
            file = root / f"{stage}-{split}-answers.jsonl"
            receipt = json.loads(file.with_suffix(".provenance.json").read_text())
            if receipt["input_sha256"] != sha(data / f"{split}-questions.jsonl"):
                raise ValueError(f"wrong questions {file}")
            pred = read_jsonl(file)
            if [r["id"] for r in refs] != [p["id"] for p in pred]:
                raise ValueError(f"misaligned {file}")
            for r, p in zip(refs, pred):
                if not 0 < len(p["token_ids"]) <= 32 or p["stop"] not in (
                    "eos",
                    "budget",
                    "pad",
                ):
                    raise ValueError(f"invalid generation {r['id']}")
                r["predictions"][stage] = {
                    k: p[k] for k in ("answer", "token_ids", "stop")
                }
            receipts[f"{stage}-{split}"] = {**receipt, "answers_sha256": sha(file)}
        rows.extend(refs)
    if len({r["id"] for r in rows}) != 64 or sum(r["primary"] for r in rows) != 60:
        raise ValueError("invalid primary/diagnostic split")
    return rows, receipts


def worksheet(rows):
    result = []
    for r in rows:
        unique = {p["answer"] for p in r["predictions"].values()}
        for answer in sorted(
            unique, key=lambda a: digest((r["id"] + "\0" + a).encode())
        ):
            if normalized(answer) != normalized(r["output"]):
                result.append(
                    dict(
                        id=r["id"],
                        input=r["input"],
                        reference=r["output"],
                        criterion=r["criterion"],
                        answer=answer,
                        meaning=None,
                        detail=None,
                        reason="",
                    )
                )
    return result


def totals(rows, stage):
    pred = [r["predictions"][stage] for r in rows]
    return dict(
        n=len(rows),
        meaning=sum(p["meaning"] for p in pred),
        detail=sum(p["detail"] for p in pred),
        exact=sum(p["exact"] for p in pred),
    )


def score(rows, reviews):
    ledger = {(x["id"], x["answer"]): x for x in reviews}
    expected = {(r["id"], r["answer"]) for r in worksheet(rows)}
    if len(ledger) != len(reviews) or set(ledger) != expected:
        raise ValueError("reviews must cover every unique non-exact answer")
    for r in rows:
        for p in r["predictions"].values():
            p["exact"] = normalized(p["answer"]) == normalized(r["output"])
            if p["exact"]:
                p.update(meaning=True, detail=True, reason="与参考译文一致。")
            else:
                entry = ledger[r["id"], p["answer"]]
                if (
                    any(type(entry[k]) is not bool for k in ("detail", "meaning"))
                    or (entry["meaning"] and not entry["detail"])
                    or not entry["reason"].strip()
                ):
                    raise ValueError(f"incomplete review {r['id']}")
                p.update({k: entry[k] for k in ("meaning", "detail", "reason")})
    primary = [r for r in rows if r["primary"]]
    stats = {}
    for stage in STAGES:
        good = Counter(r["pair"] for r in primary if r["predictions"][stage]["meaning"])
        stats[stage] = dict(
            primary=totals(primary, stage),
            both_directions=sum(v == 2 for v in good.values()),
            directions={
                d: totals([r for r in primary if r["direction"] == d], stage)
                for d in ("en-zh", "zh-en")
            },
            groups={
                g: totals([r for r in primary if r["group"] == g], stage)
                for g in GROUPS
            },
            diagnostics=totals([r for r in rows if not r["primary"]], stage),
        )
        stats[stage]["paired_changes"] = {}
        for reference in ("before", "wide"):
            stats[stage]["paired_changes"][reference] = {
                metric: {
                    "gained": sum(
                        r["predictions"][stage][metric]
                        and not r["predictions"][reference][metric]
                        for r in primary
                    ),
                    "lost": sum(
                        not r["predictions"][stage][metric]
                        and r["predictions"][reference][metric]
                        for r in primary
                    ),
                }
                for metric in ("meaning", "detail")
            }
    return stats


def report(root, rows, receipts, ledger=REVIEWS):
    freeze = json.loads(FREEZE.read_text())
    corpus = [
        json.loads(line) for line in gzip.decompress(SUBSET.read_bytes()).splitlines()
    ]
    base = receipts["before-heldout"]["model_sha256"]
    common_binary = receipts["before-heldout"]["binary_sha256"]
    training = {}
    for name in ARMS:
        stamp = json.loads((root / f"{name}-run.json").read_text())
        trained = json.loads((root / name / "training-report.json").read_text())
        h = sha(root / name / "model.bpk")
        inp = stamp["inputs"]
        if (
            inp["base_sha256"] != base
            or inp["binary_sha256"] != common_binary
            or inp["data_sha256"] != freeze["files"][name + ".jsonl"]
            or sha(root / name / "tokenizer.json") != inp["tokenizer_sha256"]
            or stamp["model_sha256"] != h
            or trained["runtime"] != "Metal"
            or trained["parameters"] != 125276928
            or trained["optimizer_steps"] != freeze["steps"]
            or trained["schedule"]
            != {"batch_size": 16, "num_epochs": 4, "learning_rate": 0.00015, "seed": 42}
        ):
            raise ValueError(f"invalid training receipt {name}")
        examples = [r["en"] for r in corpus if name in r.get("arms", [])]
        losses = [
            float(value)
            for value in re.findall(
                r"mean_batch_loss=([0-9.eE+-]+)", (root / f"{name}.log").read_text()
            )
        ]
        if len(losses) != freeze["epochs"]:
            raise ValueError(f"incomplete epoch log for {name}")
        training[name] = {
            "epoch_mean_batch_loss": losses,
            **stamp,
            **trained,
            "topic_hits": {
                k: sum(bool(re.search(v, s, re.I)) for s in examples)
                for k, v in TOPICS.items()
            },
            "form_hits": {
                k: sum(bool(re.search(v, s, re.I)) for s in examples)
                for k, v in PATTERNS.items()
            },
        }
    for stage in STAGES:
        for split in ("heldout", "reserve"):
            receipt = receipts[f"{stage}-{split}"]
            weight = base if stage == "before" else training[stage]["model_sha256"]
            if (
                receipt["model_sha256"] != weight
                or receipt["binary_sha256"] != common_binary
            ):
                raise ValueError(f"inference weights changed for {stage}")
    return dict(
        experiment="daily-v4",
        freeze=freeze,
        protocol=json.loads(PROTOCOL.read_text()),
        training=training,
        inference=receipts,
        review_sha256=sha(ledger),
        scores=score(rows, read_jsonl(ledger)),
        results=rows,
    )


def answer_page(result, raw, dest):
    def safe(s):
        return html.escape(s, quote=False).replace("|", "&#124;").replace("\n", "<br>")

    lines = [
        "# 真实日常句逐句双向翻译",
        "",
        "✓ 为整句通过，△ 为指定难点正确但整句有误，✗ 为两项均未通过。小语料、广泛采样、难点采样各训练 1,536 步。"
        f"[原答、各组逐句理由及 token ID]({os.path.relpath(raw,dest.parent)})",
        "",
    ]
    for group, label in [*GROUPS.items(), ("diagnostic", "与旧语料相近的两个诊断句对")]:
        part = [
            r
            for r in result["results"]
            if (r["group"] == group and r["primary"])
            or (group == "diagnostic" and not r["primary"])
        ]
        lines.extend(
            [
                f"## {label}",
                "",
                "| 原句 | 参考译文 | 上一版 | 384 句对 | 广泛采样 | 难点采样 | 难点组判定依据 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        credits = {}
        for r in part:
            cells = [r["input"], r["output"]]
            for stage in STAGES:
                p = r["predictions"][stage]
                cells.append(
                    ("✓ " if p["meaning"] else "△ " if p["detail"] else "✗ ")
                    + p["answer"]
                )
            cells.append(r["predictions"]["focused"]["reason"])
            lines.append("| " + " | ".join(safe(c) for c in cells) + " |")
            credits[r["pair"]] = r["credit"]
        lines.extend(
            [
                "",
                "<details><summary>本组原句署名</summary>",
                "",
                *["- " + v for v in credits.values()],
                "",
                "</details>",
                "",
            ]
        )
    dest.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reviews", type=Path, default=REVIEWS)
    parser.add_argument("--worksheet", action="store_true")
    parser.add_argument("--answers-md", type=Path)
    a = parser.parse_args()
    data, proof = inputs(a.root)
    if a.worksheet:
        jsonl(a.output, worksheet(data))
    else:
        result = report(a.root, data, proof, a.reviews)
        a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if a.answers_md:
            answer_page(result, a.output, a.answers_md)
