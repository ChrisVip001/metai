"""Time one stage of the M2 Pro run using a prebuilt Metal binary and prepared Tatoeba data."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def run(root, binary, stage, batch, epochs, lr, tokenizer="tokenizer.json"):
    data = root / "data"
    manifest = json.loads((data / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"prepared data changed: {name}")
    if stage == "evaluate":
        ids = [
            json.loads(line)["id"]
            for line in (data / "questions.jsonl").read_text().splitlines()
        ]
        for name in ("pretrain", "sft"):
            output = root / f"{name}-answers.jsonl"
            args = [
                binary,
                "--device",
                "metal",
                "eval-chat",
                "--model-dir",
                root / name,
                "--questions",
                data / "questions.jsonl",
                "--max-len",
                "32",
            ]
            if output.exists():
                raise FileExistsError(output)
            with tempfile.NamedTemporaryFile(
                dir=root, prefix=f"{name}-evaluating-", mode="w+", encoding="utf-8"
            ) as f:
                subprocess.run(args, stdout=f, check=True)
                f.seek(0)
                generated = f.read()
            if [json.loads(line)["id"] for line in generated.splitlines()] != ids:
                raise ValueError(f"incomplete {name} predictions")
            with output.open("x", encoding="utf-8") as dst:
                dst.write(generated)
            print(f"wrote {output}", flush=True)
        return
    out = root / stage
    if out.exists():
        raise FileExistsError(out)
    if stage == "sft":
        args = [
            "train-sft",
            "--model-dir",
            root / "pretrain",
            "--data-path",
            data / "sft.jsonl",
        ]
    else:
        args = [
            "pretrain",
            "--size",
            "125m",
            "--tokenizer-path",
            tokenizer,
            "--train-path",
            data / ("pilot.txt" if stage == "pilot" else "pretrain.txt"),
            "--valid-path",
            data / "valid.txt",
        ]
    args += [
        "--num-epochs",
        epochs,
        "--batch-size",
        batch,
        "--learning-rate",
        lr,
        "--seed",
        42,
        "--output-dir",
        out,
    ]
    cmd = list(map(str, [binary, "--device", "metal", *args]))
    binary_hash = hashlib.sha256()
    with binary.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            binary_hash.update(chunk)
    info = dict(
        command=cmd,
        binary_sha256=binary_hash.hexdigest(),
        data_sha256=hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest(),
    )
    print(" ".join(cmd), flush=True)
    start = time.monotonic()
    with (root / f"{stage}.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            cmd,
            env={**os.environ, "METAI_LOG_EVERY": "8"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as proc:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line.rstrip(), flush=True)
            info.update(
                exit_code=proc.wait(),
                elapsed_seconds=round(time.monotonic() - start, 3),
            )
    (root / f"{stage}-run.json").write_text(json.dumps(info, indent=2) + "\n")
    if info["exit_code"]:
        raise SystemExit(info["exit_code"])
    print(f'{stage}: {info["elapsed_seconds"]} seconds', flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("stage", choices=["pilot", "pretrain", "sft", "evaluate"])
    p.add_argument("--binary", type=Path, default=Path("target/debug/metai"))
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizer.json"),
        help="vocabulary for fresh pretraining",
    )
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.0003)
    args = p.parse_args()
    run(
        args.root,
        args.binary,
        args.stage,
        args.batch,
        args.epochs,
        args.lr,
        args.tokenizer,
    )
