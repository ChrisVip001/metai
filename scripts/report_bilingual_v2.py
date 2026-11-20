"""Join checkpoints' outputs on identical questions; measure repetition and source sensitivity."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from score_bilingual import bigrams, normalized, read_jsonl, score, sha, summary


def translation_scores(rows, stage):
    answers = [normalized(r[stage]["answer"]) for r in rows]
    gold = [bigrams(r["output"]) for r in rows]
    generated = [bigrams(a) for a in answers]
    denominator = sum(sum(c.values()) for c in gold + generated)

    def f1(shift):
        return (
            2
            * sum(
                sum((a & gold[(i + shift) % len(rows)]).values())
                for i, a in enumerate(generated)
            )
            / max(1, denominator)
        )

    return dict(
        n=len(rows),
        distinct_answers=len(set(answers)),
        most_frequent_count=Counter(answers).most_common(1)[0][1],
        aligned_bigram_f1=f1(0),
        rotated_bigram_f1=mean(f1(i) for i in range(1, len(rows))),
    )


def report(root, old, output):
    result = score(root, root / "base-and-sft-report.json")
    result["control_inputs"] = json.loads((root / "control-inputs.json").read_text())
    result["translation_control_inputs"] = json.loads(
        (root / "translation-control-inputs.json").read_text()
    )
    rows = result["questions"]
    ids = [r["id"] for r in rows]
    for label, model in (
        ("old-sft", old / "sft"),
        ("small-sft", root / "small-sft"),
        ("legacy-translation-sft", root / "legacy-translation-sft"),
    ):
        extra = read_jsonl(root / f"{label}-answers.jsonl")
        if [p["id"] for p in extra] != ids:
            raise ValueError(f"unexpected {label} predictions")
        for r, p in zip(rows, extra):
            r[label] = {k: p[k] for k in ("answer", "token_ids", "stop")}
            r[label]["exact"] = normalized(r["output"]) == normalized(p["answer"])
        result["results"][label] = summary(rows, label)
        result["runs"][label] = dict(
            training=json.loads((model / "training-report.json").read_text()),
            tokenizer_sha256=sha(model / "tokenizer.json"),
            checkpoint_sha256=sha(model / "model.bpk"),
        )
    for label in result["results"]:
        result["results"][label]["translations"] = {
            k: translation_scores([r for r in rows if r["kind"] == k], label)
            for k in ("en-zh", "zh-en")
        }
    contrasts = read_jsonl(root / "data/contrast.jsonl")
    dev = [
        r for r in read_jsonl(root / "data/attribution.jsonl") if r["split"] == "dev"
    ]
    for i, row in enumerate(contrasts):
        lang = "en" if row["kind"] == "en-zh" else "zh"
        swapped = dev[(i + 2) % len(dev)]
        if dev[i][lang] != row["input"] or swapped[lang] != row["swapped_input"]:
            raise ValueError("mismatched development attribution")
        row["credit"] = dev[i]["credit"]
        row["swapped_credit"] = swapped["credit"]
    for label in ("old-sft", "small-sft", "legacy-translation-sft", "sft"):
        observations = read_jsonl(root / f"{label}-source.jsonl")
        if [o["id"] for o in observations] != [r["id"] for r in contrasts]:
            raise ValueError("misaligned source contrasts")
        for r, o in zip(contrasts, observations):
            r[label] = {
                k: o[k]
                for k in ("matched_nll", "swapped_nll", "targets", "cache_check")
            }
        diffs = [o["swapped_nll"] - o["matched_nll"] for o in observations]
        result["results"][label]["source_test"] = dict(
            n=len(diffs),
            matched_lower=sum(d > 0 for d in diffs),
            ties=sum(d == 0 for d in diffs),
            mean_swapped_minus_matched=mean(diffs),
        )
    result["development_contrasts"] = contrasts
    for label in ("old-sft", "small-sft", "legacy-translation-sft", "sft"):
        result["runs"][label]["source_scoring"] = json.loads(
            (root / f"{label}-source.provenance.json").read_text()
        )
    for label in ("old-sft", "small-sft", "legacy-translation-sft"):
        result["runs"][label]["generation"] = json.loads(
            (root / f"{label}-answers.provenance.json").read_text()
        )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "old", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    for key, r in report(a.root, a.old, a.output)["results"].items():
        print(key, r["translations"], r.get("source_test"))
