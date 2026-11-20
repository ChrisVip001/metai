"""Replay v1 supervision and score source substitutions on v2's frozen questions."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from prepare_bilingual import jsonl, task
from score_bilingual import read_jsonl, sha


def fit_control(root, name, base, dataset, binary, stamp_name):
    model = root / name
    cmd = [
        binary,
        "--device",
        "metal",
        "train-sft",
        "--model-dir",
        base,
        "--data-path",
        dataset,
        "--output-dir",
        model,
        "--batch-size",
        "8",
        "--learning-rate",
        "0.0001",
        "--num-epochs",
        "2",
        "--seed",
        "42",
    ]
    provenance = dict(
        command=list(map(str, cmd)),
        binary_sha256=sha(binary),
        base_sha256=sha(base / "model.bpk"),
        control_data_sha256=sha(dataset),
    )
    stamp = root / stamp_name
    if model.exists():
        if not stamp.exists() or json.loads(stamp.read_text()) != provenance:
            raise ValueError(f"existing {name} has different or unknown provenance")
        return model
    with (root / f"{name}.log").open("w") as log:
        subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env={**os.environ, "METAI_LOG_EVERY": "8"},
        )
    stamp.write_text(json.dumps(provenance, indent=2) + "\n")
    return model


def only_translations(root, old):
    pairs = [
        r for r in read_jsonl(old / "data/attribution.jsonl") if r["split"] == "sft"
    ]
    translated = [task(r, ("en-zh", "zh-en")[i % 2]) for i, r in enumerate(pairs)]
    original = read_jsonl(old / "data/sft.jsonl")
    if len(original) != 256 or sum(a == b for a, b in zip(translated, original)) != 128:
        raise ValueError("the v1 ablation requires exactly 128 unchanged records")
    path = root / "legacy-translations.jsonl"
    if path.exists():
        if read_jsonl(path) != translated:
            raise ValueError("unexpected translation-only control data")
    else:
        jsonl(path, translated)
    return path


def evaluate(cmd, output, expected_ids, model_hash, input_path):
    provenance = dict(
        command=list(map(str, cmd)),
        binary_sha256=sha(cmd[0]),
        model_sha256=model_hash,
        input_sha256=sha(input_path),
    )
    stamp = output.with_suffix(".provenance.json")
    if output.exists():
        if not stamp.exists() or json.loads(stamp.read_text()) != provenance:
            raise ValueError(f"unverified existing {output}")
        rows = read_jsonl(output)
    else:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".eval-", mode="w+"
        ) as f:
            subprocess.run(cmd, stdout=f, check=True)
            f.seek(0)
            rows = [json.loads(line) for line in f]
        if [r["id"] for r in rows] != expected_ids:
            raise ValueError(f"incomplete {output}")
        stamp.write_text(json.dumps(provenance, indent=2) + "\n")
        jsonl(output, rows)
    if [r["id"] for r in rows] != expected_ids:
        raise ValueError(f"incomplete {output}")


def compare(root, old, binary, probe, task_ablation):
    data = root / "data"
    for directory in (data, old / "data"):
        manifest = json.loads((directory / "manifest.json").read_text())
        for filename, expected in manifest["files"].items():
            if sha(directory / filename) != expected:
                raise ValueError(f"changed {directory/filename}")
    if task_ablation:
        label = "legacy-translation-sft"
        model = fit_control(
            root,
            label,
            old / "pretrain",
            only_translations(root, old),
            binary,
            "translation-control-inputs.json",
        )
        participants = [(label, model)]
    else:
        control = fit_control(
            root,
            "small-sft",
            root / "pretrain",
            old / "data/sft.jsonl",
            binary,
            "control-inputs.json",
        )
        participants = [
            ("old-sft", old / "sft"),
            ("small-sft", control),
            ("sft", root / "sft"),
        ]
    questions = data / "questions.jsonl"
    contrast = data / "contrast.jsonl"
    test_ids = [r["id"] for r in read_jsonl(questions)]
    dev_ids = [r["id"] for r in read_jsonl(contrast)]
    for label, model in participants:
        model_hash = sha(model / "model.bpk")
        if label != "sft":
            cmd = [
                binary,
                "--device",
                "metal",
                "eval-chat",
                "--model-dir",
                model,
                "--questions",
                questions,
                "--max-len",
                "32",
            ]
            evaluate(
                cmd, root / f"{label}-answers.jsonl", test_ids, model_hash, questions
            )
        cmd = [probe, "--model-dir", model, "--contrast", contrast]
        evaluate(cmd, root / f"{label}-source.jsonl", dev_ids, model_hash, contrast)
        print(f"completed {label}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "old", "binary", "probe"):
        p.add_argument(name, type=Path)
    p.add_argument(
        "--task-ablation",
        action="store_true",
        help="replace the v1 language-label tasks with translations on the same v1 base",
    )
    a = p.parse_args()
    compare(a.root, a.old, a.binary, a.probe, a.task_ablation)
