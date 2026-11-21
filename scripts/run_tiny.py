"""Run the CPU course via the public Rust CLI. Each invocation owns a fresh directory."""
import argparse
import json
from itertools import product
from math import sin, tau
import os
import subprocess
import time
from pathlib import Path

from clean_corpus import prepare

ROOT = Path(__file__).resolve().parents[1]


def generate_inputs(folder):
    folder.mkdir()

    def save(name, rows):
        (folder / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    numbers = ("zero", "one", "two", "three", "four")
    corpus = [
        f"{numbers[a]} plus {numbers[b]} is {numbers[a + b]}."
        for a, b in product((1, 2), repeat=2)
    ]
    corpus += [f"{color} is a color." for color in ("red", "blue")] + [
        f"the {item} is {color}." for item, color in (("sky", "blue"), ("apple", "red"))
    ]
    corpus += [
        "a tool can add two numbers.",
        "give the sum of one and one.",
        "what is two plus two?",
        "the answer is four.",
        "one   plus one is two.",
    ]
    (folder / "corpus.utf8").write_text("\n".join(corpus) + "\n", encoding="utf-8")
    save(
        "sft",
        [
            dict(instruction=f"{a}+{b}?", output=str(a + b))
            for a, b in product((1, 2), repeat=2)
        ],
    )
    save(
        "preferences",
        [
            dict(instruction=f"{a}+{a}?", chosen=str(2 * a), rejected=str(2 * a + 5))
            for a in (1, 2)
        ],
    )
    for split, pairs in (
        (
            "train",
            zip(("apple", "sky", "rose", "sea", "flower", "ink"), ("red", "blue") * 3),
        ),
        ("cal", (("ink", "red"), ("paint", "blue"), ("paint", "red"), ("cup", "blue"))),
        (
            "test",
            (("cup", "red"), ("paper", "blue"), ("paper", "red"), ("flower", "blue")),
        ),
    ):
        save(
            f"decision-{split}",
            [dict(text=f"{color} {noun}", label=color) for noun, color in pairs],
        )
    images = [
        [int((i % 4 if vertical else i // 4) < 2) for i in range(16)]
        for vertical in (False, True)
    ]
    save(
        "image",
        [
            dict(image=image, question="Stripe?", answer=answer)
            for image, answer in zip(images, ("horizontal", "vertical"))
        ],
    )
    waves = [[round(sin(tau * freq * i / 16), 5) for i in range(16)] for freq in (1, 2)]
    save(
        "audiovisual",
        [
            dict(
                image=image,
                audio=wave,
                question="Match?",
                answer="yes" if i == j else "no",
            )
            for i, image in enumerate(images)
            for j, wave in enumerate(waves)
        ],
    )



def run(output, seed):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixtures = output / "fixtures"
    generate_inputs(fixtures)
    info = dict(seed=seed, backend="Burn Flex CPU",
                rustc=subprocess.check_output(["rustc", "--version"], text=True).strip(), stages=[])
    info["data"] = prepare(fixtures / "corpus.utf8", output / "data", seed)
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
    stage("sft", "train-sft", "--model-dir", "base", "--data-path", fixtures / "sft.jsonl",
          "--output-dir", "sft", "--num-epochs", 2, *schedule)
    for method, data in [("lora", "sft.jsonl"), ("dpo", "preferences.jsonl"), ("ppo", "sft.jsonl"),
                         ("grpo", "sft.jsonl"), ("cispo", "sft.jsonl"), ("thinking", "sft.jsonl")]:
        stage(method, f"train-{method}", "--model-dir", "sft", "--data-path", fixtures / data,
              "--output-dir", method, *schedule)
    stage("ai-feedback", "train-grpo", "--model-dir", "sft", "--judge-dir", "base",
          "--data-path", fixtures / "sft.jsonl", "--output-dir", "rlaif", "--max-new-tokens", 2, *schedule)
    stage("distill", "distill", "--model-dir", "sft", "--data-path", "data/train.txt", "--output-dir", "student", *schedule)
    stage("generate", "generate", "--model-dir", "student", "--prompt", "1+1?", "--chat", "--max-len", 4)
    stage("think", "think", "--model-dir", "thinking", "--prompt", "1+1?", "--max-len", 4)
    stage("agent", "train-agent", "--model-dir", "sft", "--output-dir", "agent", "--num-epochs", 75, "--learning-rate", 0.03, *schedule)
    stage("tool-sft", "play-agent", "--model-dir", "agent", "--case", 3, "--imitation")
    stage("agent-play", "play-agent", "--model-dir", "agent", "--case", 3)
    stage("decision", "train-decision", "--model-dir", "sft", "--output-dir", "decision",
          "--data-path", fixtures / "decision-train.jsonl", "--calibration-path", fixtures / "decision-cal.jsonl",
          "--test-path", fixtures / "decision-test.jsonl", "--num-epochs", 16, "--learning-rate", 0.01, *schedule)
    stage("decide", "decide", "--model-dir", "decision", "--text", "red cup")
    for dataset, flags in [("image", []), ("audiovisual", ["--audio"])]:
        stage(dataset, "train-multimodal", "--model-dir", "sft", "--data-path", fixtures / f"{dataset}.jsonl",
              "--output-dir", dataset, *flags, *schedule)
        stage(f"{dataset}-generate", "generate-multimodal", "--model-dir", dataset,
              "--data-path", fixtures / f"{dataset}.jsonl", "--max-len", 2)
    for kind in ["vision", "omni", "linear", "diffusion"]:
        stage(kind, "lab", kind, "--output-dir", kind, "--num-epochs", 12, "--learning-rate", 0.01, *schedule)
    print(output / "run.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.output, args.seed)
