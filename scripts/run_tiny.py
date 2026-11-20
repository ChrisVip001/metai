"""Run the CPU course via the public Rust CLI. Each invocation owns a fresh directory."""
import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from clean_corpus import prepare

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "examples" / "tiny"


def run(output, seed):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    info = dict(seed=seed, backend="Burn Flex CPU", machine=platform.machine(),
                rustc=subprocess.check_output(["rustc", "--version"], text=True).strip(), stages=[])
    info["data"] = prepare(FIXTURES / "corpus.utf8", output / "data", seed)
    build = subprocess.check_output(["cargo", "build", "--locked", "--message-format=json"], cwd=ROOT, text=True)
    executable = next(Path(row["executable"]) for line in build.splitlines()
                      if (row := json.loads(line)).get("executable") and row["target"]["name"] == "metai")
    log = output / "logs"
    log.mkdir()
    def stage(name, *args):
        command = [os.fspath(executable), *map(str, args)]
        start = time.monotonic()
        with (log / f"{name}.log").open("w", encoding="utf-8") as stream:
            proc = subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT)
        info["stages"].append(dict(name=name, args=command[1:], exit_code=proc.returncode,
                                   seconds=round(time.monotonic()-start, 3)))
        (output / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"{name} failed. See {log / (name + '.log')}")
        print(f"{name}  {info['stages'][-1]['seconds']}s", flush=True)

    schedule = ["--seed", seed]
    stage("pretrain", "pretrain", "--train-path", "data/train.txt", "--valid-path", "data/valid.txt",
          "--output-dir", "base", "--checkpoint-dir", "epochs", "--num-epochs", 3, *schedule)
    stage("resume", "resume-pretrain", "--model-dir", "epochs/epoch-2", "--train-path", "data/train.txt",
          "--valid-path", "data/valid.txt", "--output-dir", "resumed")
    stage("sft", "train-sft", "--model-dir", "base", "--data-path", FIXTURES / "sft.jsonl",
          "--output-dir", "sft", "--num-epochs", 2, *schedule)
    for method, data in [("lora", "sft.jsonl"), ("dpo", "preferences.jsonl"), ("ppo", "sft.jsonl"),
                         ("grpo", "sft.jsonl"), ("cispo", "sft.jsonl"), ("thinking", "sft.jsonl")]:
        stage(method, f"train-{method}", "--model-dir", "sft", "--data-path", FIXTURES / data,
              "--output-dir", method, *schedule)
    stage("ai-feedback", "train-grpo", "--model-dir", "sft", "--judge-dir", "base",
          "--data-path", FIXTURES / "sft.jsonl", "--output-dir", "rlaif", "--max-new-tokens", 2, *schedule)
    stage("distill", "distill", "--model-dir", "sft", "--data-path", "data/train.txt", "--output-dir", "student", *schedule)
    stage("generate", "generate", "--model-dir", "student", "--prompt", "1+1?", "--chat", "--max-len", 4)
    stage("think", "think", "--model-dir", "thinking", "--prompt", "1+1?", "--max-len", 4)
    stage("agent", "train-agent", "--model-dir", "sft", "--output-dir", "agent", "--num-epochs", 75, "--learning-rate", 0.03, *schedule)
    stage("tool-sft", "play-agent", "--model-dir", "agent", "--case", 3, "--imitation")
    stage("agent-play", "play-agent", "--model-dir", "agent", "--case", 3)
    stage("decision", "train-decision", "--model-dir", "sft", "--output-dir", "decision",
          "--data-path", FIXTURES / "decision-train.jsonl", "--calibration-path", FIXTURES / "decision-cal.jsonl",
          "--test-path", FIXTURES / "decision-test.jsonl", "--num-epochs", 16, "--learning-rate", 0.01, *schedule)
    stage("decide", "decide", "--model-dir", "decision", "--text", "red cup")
    for dataset, flags in [("image", []), ("audiovisual", ["--audio"])]:
        stage(dataset, "train-multimodal", "--model-dir", "sft", "--data-path", FIXTURES / f"{dataset}.jsonl",
              "--output-dir", dataset, *flags, *schedule)
        stage(f"{dataset}-generate", "generate-multimodal", "--model-dir", dataset,
              "--data-path", FIXTURES / f"{dataset}.jsonl", "--max-len", 2)
    for kind in ["vision", "omni", "linear", "diffusion"]:
        stage(kind, "lab", kind, "--output-dir", kind, "--num-epochs", 12, "--learning-rate", 0.01, *schedule)
    print(output / "run.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.output, args.seed)
