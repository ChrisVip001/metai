"""在相同留出集上对照第一课和改进后的决策、视听模型。"""

import argparse
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Complete noun groups are assigned before pairing each noun with both colors.
NOUNS = {
    "train": "apple sky rose sea flower ink stone chair shirt balloon leaf drum book kite bowl wagon",
    "cal": "boat spoon clock sock",
    "dev": "button basket plate pencil scarf cabin",
    "test": "lantern kettle ribbon jacket hammer bottle carpet ticket bucket guitar helmet napkin pillow sandal candle violin",
}
# One group shares two images and two waves across all four combinations.
SCENES = {"train": (1100, 8), "dev": (2100, 4), "test": (3100, 12)}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def audiovisual(seed):
    rng = random.Random(seed)
    images = []
    for vertical in [False, True]:
        dark, bright = rng.uniform(0.02, 0.16), rng.uniform(0.80, 0.98)
        pixels = []
        for i in range(16):
            index = i % 4 if vertical else i // 4
            pixels.append(round((bright if index < 2 else dark) + rng.uniform(-0.04, 0.04), 5))
        images.append(pixels)
    waves = []
    for frequency in [1, 2]:
        phase, amplitude = rng.uniform(-0.3, 0.3), rng.uniform(0.7, 1.0)
        waves.append(
            [
                round(
                    amplitude * math.sin(frequency * math.tau * i / 16 + phase)
                    + rng.uniform(-0.04, 0.04),
                    5,
                )
                for i in range(16)
            ]
        )
    return [
        dict(image=image, audio=wave, question="Match?", answer="yes" if i == j else "no")
        for i, image in enumerate(images)
        for j, wave in enumerate(waves)
    ]


def prepare(data):
    data.mkdir()
    nouns, all_images, all_waves = set(), set(), set()
    for split, text in NOUNS.items():
        words = text.split()
        assert len(set(words)) == len(words) and not nouns.intersection(words)
        nouns.update(words)
        write_rows(
            data / f"decision-{split}.jsonl",
            [dict(text=f"{c} {w}", label=c) for w in words for c in ("red", "blue")],
        )
    for split, (start, groups) in SCENES.items():
        rows = [row for i in range(groups) for row in audiovisual(start + i)]
        images = {tuple(row["image"]) for row in rows}
        waves = {tuple(row["audio"]) for row in rows}
        assert len(images) == len(waves) == groups * 2
        assert images.isdisjoint(all_images) and waves.isdisjoint(all_waves)
        all_images.update(images)
        all_waves.update(waves)
        write_rows(data / f"av-{split}.jsonl", rows)


def decision_eval(binary, model, rows):
    predictions = []
    nll = brier = 0.0
    for row in rows:
        out = subprocess.check_output(
            [binary, "decide", "--model-dir", model, "--text", row["text"]], text=True
        )
        result = json.loads(out)
        p = dict(zip(result["labels"], result["probabilities"]))
        nll -= math.log(max(p[row["label"]], 1.0e-300))
        brier += sum((v - (k == row["label"])) ** 2 for k, v in p.items())
        predictions.append(dict(**row, predicted=result["label"], p_red=p["red"]))
    n = len(rows)
    correct = sum(row["label"] == row["predicted"] for row in predictions)
    return dict(n=n, correct=correct, nll=nll / n, brier=brier / n, predictions=predictions)


def av_eval(binary, model, inputs, gold, limit, eos):
    out = subprocess.check_output(
        [
            binary,
            "generate-multimodal",
            "--model-dir",
            model,
            "--data-path",
            inputs,
            "--max-len",
            str(limit),
        ],
        text=True,
    )
    predictions = [json.loads(line) for line in out.split("\n") if line.strip()]
    if len(predictions) != len(gold):
        raise ValueError("生成条数与参考答案不符")
    detail = []
    for pred, row in zip(predictions, gold):
        if row["question"] != pred["question"]:
            raise ValueError("生成问题与原始输入不匹配")
        detail.append(dict(gold=row["answer"], raw=pred["answer"], ids=pred["token_ids"]))
    matches = [row["raw"] == row["gold"] for row in detail]
    return dict(
        n=len(detail),
        correct=sum(matches),
        replacement=sum("\ufffd" in r["raw"] for r in detail),
        eos=sum(bool(r["ids"]) and r["ids"][-1] == eos for r in detail),
        quartets_correct=sum(all(matches[i : i + 4]) for i in range(0, len(matches), 4)),
        predictions=detail,
    )


def run(course, out, development):
    course = course.resolve()
    out = out.resolve()
    fixtures = course / "fixtures"
    source = json.loads((course / "run.json").read_text())
    for name in ("sft", "decision", "audiovisual"):
        if not any(s["name"] == name and s["exit_code"] == 0 for s in source["stages"]):
            raise ValueError(f"第一课缺少成功的 {name} 阶段")
    out.mkdir(parents=True, exist_ok=False)
    data = out / "data"
    prepare(data)
    build = subprocess.check_output(
        ["cargo", "build", "--locked", "--message-format=json"], cwd=ROOT, text=True
    )
    binary = next(
        Path(r["executable"])
        for line in build.splitlines()
        if (r := json.loads(line)).get("executable") and r["target"]["name"] == "metai"
    )
    split = "dev" if development else "test"
    seed = source["seed"]
    report = dict(
        split=split,
        seed=seed,
        backend=source["backend"],
        rustc=source["rustc"],
        data_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob("*.jsonl"))
        },
        decision={},
        audiovisual={},
        training_commands={},
    )

    def train(name, command, *args):
        argv = [command, "--model-dir", course / "sft", "--output-dir", out / name, "--seed", seed, *args]
        report["training_commands"][name] = [
            (
                str(x).replace(str(course), "$COURSE").replace(str(out), "$RUN")
                if isinstance(x, Path)
                else str(x)
            )
            for x in argv
        ]
        print(f"训练 {name}", flush=True)
        with (out / f"{name}.log").open("w") as log:
            subprocess.run([binary, *map(str, argv)], stdout=log, check=True)

    for pooling in ["last", "mean"]:
        train(
            "decision-" + pooling,
            "train-decision",
            "--pooling",
            pooling,
            "--data-path",
            data / "decision-train.jsonl",
            "--calibration-path",
            data / "decision-cal.jsonl",
            "--test-path",
            data / "decision-dev.jsonl",
            "--num-epochs",
            40,
            "--batch-size",
            4,
            "--learning-rate",
            0.015,
        )
    for name, flags in [("av-prefix", []), ("av-readout", ["--train-readout"])]:
        train(
            name,
            "train-multimodal",
            "--audio",
            "--continuation-targets",
            *flags,
            "--data-path",
            data / "av-train.jsonl",
            "--num-epochs",
            40,
            "--batch-size",
            8,
            "--learning-rate",
            0.008,
        )

    # Holdout labels are only consulted after the four checkpoints have been written.
    print(f"评估 {split} 集", flush=True)
    decision = read_rows(data / f"decision-{split}.jsonl")
    historical = [
        row
        for split in ("train", "cal", "test")
        for row in read_rows(fixtures / f"decision-{split}.jsonl")
    ]
    assert {r["text"] for r in historical}.isdisjoint(r["text"] for r in decision)
    probes = [dict(text=f"{c} cup", label=c) for c in ["red", "blue"]]
    for method, model in [
        ("baseline", course / "decision"),
        ("last", out / "decision-last"),
        ("mean", out / "decision-mean"),
    ]:
        scores = decision_eval(binary, model, decision)
        scores["cup_probes"] = decision_eval(binary, model, probes)["predictions"]
        scores["temperature"] = json.loads((model / "schema.json").read_text())["temperature"]
        report["decision"][method] = scores

    av = read_rows(data / f"av-{split}.jsonl")
    old_av = read_rows(fixtures / "audiovisual.jsonl")
    for field in ("image", "audio"):
        assert {tuple(row[field]) for row in av}.isdisjoint(tuple(row[field]) for row in old_av)
    inputs = data / f"av-{split}-inputs.jsonl"
    write_rows(inputs, [{k: v for k, v in row.items() if k != "answer"} for row in av])
    vocab = json.loads((course / "sft/tokenizer.json").read_text())["model"]["vocab"]
    for name, model in [
        ("baseline", course / "audiovisual"),
        ("prefix", out / "av-prefix"),
        ("readout", out / "av-readout"),
    ]:
        report["audiovisual"][name] = av_eval(binary, model, inputs, av, 6, vocab["</s>"])
    old_inputs = data / "av-original-inputs.jsonl"
    write_rows(old_inputs, [{k: v for k, v in row.items() if k != "answer"} for row in old_av])
    report["original_av_budget2"] = av_eval(
        binary, course / "audiovisual", old_inputs, old_av, 2, vocab["</s>"]
    )
    report["example_quartet"] = av[:4]
    write_json(out / "report.json", report)
    for name in ("decision", "audiovisual"):
        print(name, {k: f"{v['correct']}/{v['n']}" for k, v in report[name].items()})
    print(out / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", type=Path, required=True, help="run_tiny.py 的输出")
    parser.add_argument("--output", type=Path, required=True, help="新的实验目录")
    parser.add_argument("--development", action="store_true", help="只评估开发集")
    args = parser.parse_args()
    run(args.course, args.output, args.development)
