"""Run the predeclared translation experiments sequentially on one Metal device."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from compare_bilingual_runs import evaluate
from prepare_translation import ARMS, FREEZE
from score_bilingual import read_jsonl, sha


def fit(root, name, base, binary, *, schedule=None):
    batch, epochs = schedule or ((8, 80) if name == "memory" else (16, 16))
    dataset = root / "data" / f"{name}.jsonl"
    model = root / name
    cmd = list(
        map(
            str,
            [
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
                batch,
                "--num-epochs",
                epochs,
                "--learning-rate",
                "0.00015",
                "--seed",
                42,
            ],
        )
    )
    inputs = dict(
        command=cmd,
        binary_sha256=sha(binary),
        base_sha256=sha(base / "model.bpk"),
        tokenizer_sha256=sha(base / "tokenizer.json"),
        data_sha256=sha(dataset),
    )
    stamp = root / f"{name}-run.json"
    if stamp.exists():
        recorded = json.loads(stamp.read_text())
        if recorded["inputs"] == inputs and recorded["model_sha256"] == sha(
            model / "model.bpk"
        ):
            return model
        raise ValueError(f"training provenance mismatch for {name}")
    if model.exists():
        raise ValueError(
            f"incomplete training output {model}; use a fresh run directory"
        )
    start = time.monotonic()
    with (root / f"{name}.log").open("w") as log:
        subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env={**os.environ, "METAI_LOG_EVERY": "16"},
        )
    stamp.write_text(
        json.dumps(
            dict(
                inputs=inputs,
                elapsed_seconds=round(time.monotonic() - start, 2),
                model_sha256=sha(model / "model.bpk"),
            ),
            indent=2,
        )
        + "\n"
    )
    return model


def decode(root, binary, name, model, split):
    q = root / "data" / f"{split}-questions.jsonl"
    cmd = [
        binary,
        "--device",
        "metal",
        "eval-chat",
        "--model-dir",
        model,
        "--questions",
        q,
        "--max-len",
        "32",
    ]
    evaluate(
        cmd,
        root / f"{name}-{split}-answers.jsonl",
        [r["id"] for r in read_jsonl(q)],
        sha(model / "model.bpk"),
        q,
    )


def run(root, base, binary):
    manifest = json.loads((root / "data/manifest.json").read_text())
    if manifest != json.loads(FREEZE.read_text()):
        raise ValueError("prepared data differs from the frozen experiment")
    for filename, expected in manifest.items():
        if filename.endswith(".jsonl") and sha(root / "data" / filename) != expected:
            raise ValueError(f"changed {filename}")
    small = fit(root, "memory", base, binary)
    decode(root, binary, "memory", small, "memory")
    decode(root, binary, "baseline", base, "memory")
    print("completed memory experiment", flush=True)
    for name in (*ARMS, "mixed"):
        print(f"training {name}", flush=True)
        fit(root, name, base, binary)
    for name, model in [
        ("baseline", base),
        ("memory", small),
        *((n, root / n) for n in (*ARMS, "mixed")),
    ]:
        decode(root, binary, name, model, "heldout")
        print(f"scored {name}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "base", "binary"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    run(a.root, a.base, a.binary)
