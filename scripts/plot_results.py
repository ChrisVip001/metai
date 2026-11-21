"""Plot the retained weight tiers using only memory capacity and core counts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from run_size_tiers import MIB, TIERS

FONTS = (
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
)


def draw(summary, output):
    rows = sorted(summary["tiers"], key=lambda row: row["cap_mib"])
    if [row["cap_mib"] for row in rows] != list(TIERS):
        raise ValueError("expected measurements for 64, 128 and 256 MiB")
    for row in rows:
        if not 0 < row["weight_bytes"] <= row["cap_mib"] * MIB:
            raise ValueError("weight file exceeds its tier")

    hardware = (
        f"{int(summary['memory_gb'])} GB 统一内存  ·  "
        f"{int(summary['cpu_cores'])} 核 CPU  ·  {int(summary['gpu_cores'])} 核 GPU"
    )
    labels = [f"{row['cap_mib']} MiB" for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), facecolor="#f6f9fb")
    fig.subplots_adjust(top=0.70, bottom=0.18, left=0.085, right=0.945, wspace=0.3)
    fig.text(
        0.065, 0.905, "FP32 权重档位实测", fontsize=23, weight="bold", color="#153245"
    )
    fig.text(0.067, 0.81, hardware, fontsize=12, color="#486375")

    metrics = (
        ("预训练与 SFT 用时", "分钟", [row["seconds"] / 60 for row in rows], "%.1f"),
        (
            "训练进程峰值内存",
            "GiB",
            [row["peak_bytes"] / 2**30 for row in rows],
            "%.2f",
        ),
    )
    for ax, (title, unit, values, fmt) in zip(axes, metrics):
        bars = ax.bar(labels, values, color="#087c78", width=0.50)
        ax.set_title(title, loc="left", pad=12, weight="bold")
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values) * 1.19)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d8e3e8")
        ax.spines[["top", "right"]].set_visible(False)
        ax.bar_label(bars, fmt=fmt, padding=4, color="#153245")

    weights = "   ·   ".join(f"{row['weight_bytes'] / MIB:.2f}" for row in rows)
    fig.text(0.067, 0.065, f"实际权重（MiB）   {weights}", fontsize=11, color="#486375")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, metadata={"Title": "FP32 weight tiers"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="aggregate benchmark JSON")
    parser.add_argument("output", type=Path, help="output PNG")
    parser.add_argument("--font", type=Path, help="optional CJK font")
    args = parser.parse_args()
    names = FONTS
    if args.font:
        font_manager.fontManager.addfont(args.font)
        names = (font_manager.FontProperties(fname=args.font).get_name(), *names)
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in names if name in available), None)
    if not font:
        parser.error("use --font to supply a font with Chinese glyphs")
    plt.rcParams.update(
        {"font.family": [font, "DejaVu Sans"], "axes.unicode_minus": False}
    )
    draw(json.loads(args.summary.read_text()), args.output)


if __name__ == "__main__":
    main()
