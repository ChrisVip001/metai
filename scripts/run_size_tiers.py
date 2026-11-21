"""Create and verify the 64, 128, and 256 MiB FP32 model configurations."""

import argparse
import json
from pathlib import Path

MIB = 1 << 20
# Hidden width, layers, MLP width.
TIERS = {64: (384, 4, 624), 128: (576, 6, 928), 256: (768, 8, 1616)}


def config_for(cap):
    width, layers, mlp = TIERS[cap]
    return dict(
        vocab_size=16000,
        hidden_dim=width,
        num_layers=layers,
        num_heads=width // 64,
        num_kv_heads=width // 192,
        mlp_dim=mlp,
        max_seq_len=2048,
        dropout=0.0,
        num_experts=1,
        active_experts=1,
        num_shared_experts=0,
        norm="RmsNorm",
        router="SoftmaxTopK",
        sparse=None,
        attn="Gqa",
    )


def verify_weight(model, cap):
    if json.loads((model / "config.json").read_text()) != config_for(cap):
        raise ValueError(f"configuration does not match {cap} MiB: {model}")
    receipt = json.loads((model / "training-report.json").read_text())
    weight_bytes = (model / "model.bpk").stat().st_size
    overhead = weight_bytes - receipt["parameters"] * 4
    if (
        not 0 <= overhead < MIB // 16
        or not cap * MIB * 99 <= weight_bytes * 100 <= cap * MIB * 100
    ):
        raise ValueError(f"invalid FP32 file size for {cap} MiB: {weight_bytes} bytes")
    if not (model / "tokenizer.json").is_file():
        raise ValueError(f"missing tokenizer: {model}")
    return dict(
        cap_mib=cap, weight_bytes=weight_bytes, parameters=receipt["parameters"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config", help="write a preset to stdout")
    check = commands.add_parser("check", help="verify a saved checkpoint")
    for cmd in (config, check):
        cmd.add_argument("cap", type=int, choices=tuple(TIERS))
    check.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    result = (
        config_for(args.cap)
        if args.command == "config"
        else verify_weight(args.model_dir, args.cap)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
