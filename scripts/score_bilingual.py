"""Score the frozen M2 bilingual benchmark. Only this offline step reads the references."""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def normalized(value):
    return unicodedata.normalize("NFKC", value).strip()


def bigrams(s):
    chars = normalized(s)
    return Counter(zip(chars, chars[1:]))


def summary(rows, stage):
    def group(name):
        return [r for r in rows if r["kind"] in name]

    groups = {
        "en-zh": ["en-zh"],
        "zh-en": ["zh-en"],
        "language": ["lang-en", "lang-zh"],
        "lang-en": ["lang-en"],
        "lang-zh": ["lang-zh"],
        "transfer": ["arithmetic", "knowledge", "reading", "copy"],
    }
    scores = {}
    for label, kinds in groups.items():
        part = group(kinds)
        item = {"exact": sum(r[stage]["exact"] for r in part), "n": len(part)}
        if label in ("en-zh", "zh-en"):
            matches = predicted = reference = 0
            for r in part:

                a, b = bigrams(r[stage]["answer"]), bigrams(r["output"])
                matches += sum((a & b).values())
                predicted += sum(a.values())
                reference += sum(b.values())
            item["char_bigram_micro_f1"] = 2 * matches / max(1, predicted + reference)
        scores[label] = item
    scores["stop"] = dict(Counter(r[stage]["stop"] for r in rows))
    scores["empty_after_strip"] = sum(not normalized(r[stage]["answer"]) for r in rows)
    scores["contains_replacement_character"] = sum(
        "�" in r[stage]["answer"] for r in rows
    )
    scores["language_answers"] = dict(
        Counter(normalized(r[stage]["answer"]) for r in group(groups["language"]))
    )
    return scores


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(buf)
    return digest.hexdigest()


def score(root, dest):
    data = root / "data"
    manifest = json.loads((data / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(data / name) != expected:
            raise ValueError(f"changed prepared file: {name}")
    rows = read_jsonl(data / "evaluation.jsonl")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate benchmark IDs")
    report = {
        "dataset": manifest,
        "model_config": json.loads((root / "pretrain/config.json").read_text()),
        "decode": {
            "strategy": "greedy",
            "max_new_tokens": 32,
            "prompt": "chat_prompt(instruction, input)",
        },
        "runs": {},
        "results": {},
    }
    for stage in ("pretrain", "sft"):
        config = json.loads((root / stage / "config.json").read_text())
        if config != report["model_config"]:
            raise ValueError("model configuration changed between stages")
        tokenizer = json.loads((root / stage / "tokenizer.json").read_text())
        vocab = tokenizer["model"]["vocab"]
        outputs = read_jsonl(root / f"{stage}-answers.jsonl")
        if [r["id"] for r in outputs] != ids:
            raise ValueError(f"{stage} has missing, reordered or duplicate answers")
        for ref, pred in zip(rows, outputs):
            if not 0 < len(pred["token_ids"]) <= 32 or (
                pred["stop"] == "budget" and len(pred["token_ids"]) != 32
            ):
                raise ValueError(f"unexpected generation budget for {pred['id']}")
            last = pred["token_ids"][-1]
            expected_stop = {vocab["</s>"]: "eos", vocab["<pad>"]: "pad"}.get(
                last, "budget"
            )
            if pred["stop"] != expected_stop or any(
                i < 0 or i >= config["vocab_size"] for i in pred["token_ids"]
            ):
                raise ValueError(f"invalid stop reason or token ID for {pred['id']}")
            ref[stage] = {k: pred[k] for k in ("answer", "stop", "token_ids")}
            ref[stage]["exact"] = normalized(pred["answer"]) == normalized(
                ref["output"]
            )
        report["results"][stage] = summary(rows, stage)
        run = json.loads((root / f"{stage}-run.json").read_text())
        if run["exit_code"] != 0:
            raise ValueError(f"{stage} failed")
        report["runs"][stage] = dict(
            **run,
            training=json.loads((root / stage / "training-report.json").read_text()),
            epoch_mean_batch_losses=[
                float(v)
                for v in re.findall(
                    r"mean_batch_loss=([0-9.]+)", (root / f"{stage}.log").read_text()
                )
            ],
            logged_batches=[
                {"epoch": int(epoch), "step": int(step), "nll": float(value)}
                for epoch, step, value in re.findall(
                    r"epoch=(\d+) step=(\d+)/\d+ batch_nll=([0-9.]+)",
                    (root / f"{stage}.log").read_text(),
                )
            ],
            checkpoint_sha256=sha(root / stage / "model.bpk"),
            tokenizer_sha256=sha(root / stage / "tokenizer.json"),
        )
    if (
        report["runs"]["pretrain"]["tokenizer_sha256"]
        != report["runs"]["sft"]["tokenizer_sha256"]
    ):
        raise ValueError("tokenizer changed between stages")
    report["questions"] = rows
    dest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    print(json.dumps(score(a.root, a.report)["results"], ensure_ascii=False, indent=2))
