"""从第一课的运行目录或留出实验的 report.json 生成课程效果图。"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter


INK, MUTED, TEAL, ORANGE = "#183044", "#475e6d", "#087c78", "#a14c21"
GRID, PAPER = "#d8e3e8", "#f7fafb"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_arg(stages, name, flag):
    args = stages[name]["args"]
    return args[args.index(flag) + 1]


def heading(fig, title, subtitle):
    fig.text(.045, .93, title, size=21, weight="bold", color=INK)
    fig.text(.045, .856, subtitle, size=12, color=MUTED)


def draw_training(run, output, info, stages):
    log = (run / "logs/pretrain.log").read_text(encoding="utf-8")
    epochs = re.findall(r"^epoch=(\d+) examples=\d+ mean_batch_loss=([\d.eE+-]+)", log, re.M)
    tokens, validation = re.search(r"valid_tokens=(\d+) valid_nll=([\d.eE+-]+)", log).groups()
    x = [int(epoch) for epoch, _ in epochs]
    y = [float(loss) for _, loss in epochs]
    prediction = read_json(run / "logs/decide.log")
    evaluation = read_json(run / "decision/evaluation.json")["test_after"]
    prompt = stage_arg(stages, "decide", "--text")
    test_rows = read_rows(Path(stage_arg(stages, "decision", "--test-path")))
    truth = next(row["label"] for row in test_rows if row["text"] == prompt)

    fig, (loss, probs) = plt.subplots(1, 2, figsize=(11.6, 4.5), facecolor=PAPER)
    fig.subplots_adjust(left=.098, right=.97, bottom=.24, top=.67, wspace=.4)
    heading(fig, "预训练损失与决策概率", f'{info["backend"]}    ·    seed {info["seed"]}    ·    {info["data"]["train"]} 条训练文本')
    loss.set_title("预训练  /  每轮平均批损失", loc="left")
    loss.plot(x, y, color=TEAL, linewidth=2.8, marker="o", markersize=8)
    for epoch, value in zip(x, y):
        loss.annotate(f"{value:.4f}", (epoch, value), xytext=(0, 10), textcoords="offset points", ha="center", color=TEAL)
    loss.set_xticks(x)
    loss.set_xlabel("训练轮次")
    loss.margins(x=.17, y=.27)
    loss.grid(axis="y", color=GRID)
    loss.text(0, -.39, f"末轮验证 NLL  {float(validation):.4f}   ·   {tokens} 个有效 token", transform=loss.transAxes, color=MUTED)

    probs.set_title(f"决策  /  输入“{prompt}”", loc="left")
    bars = probs.barh(prediction["labels"], prediction["probabilities"], height=.48,
                      color=[TEAL if label == truth else ORANGE for label in prediction["labels"]])
    for bar, value in zip(bars, prediction["probabilities"]):
        probs.text(value + .02, bar.get_y() + bar.get_height()/2, f"{value:.1%}", va="center", color=INK)
    probs.invert_yaxis()
    probs.set_xlim(0, 1)
    probs.set_xticks([0, .25, .5, .75, 1])
    probs.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    probs.set_xlabel("温度校准后的标签概率")
    probs.grid(axis="x", color=GRID)
    correct = sum(row[i] for i, row in enumerate(evaluation["confusion_true_by_predicted"]))
    probs.text(0, -.39, f'预测 {prediction["label"]}   ·   真值 {truth}   ·   测试集答对 {correct}/{evaluation["n"]}', transform=probs.transAxes, color=MUTED)
    fig.savefig(output / "cpu-training-decision.png", dpi=160)
    plt.close(fig)


def draw_multimodal(run, output, info, stages):
    rows = read_rows(Path(stage_arg(stages, "audiovisual-generate", "--data-path")))
    generated = read_rows(run / "logs/audiovisual-generate.log")
    draw_examples(rows, generated, info["seed"], output / "multimodal-generation.png", "训练样本")


def draw_examples(rows, generated, seed, filename, sample_type):
    if len(rows) != len(generated) or not rows:
        raise ValueError("视听样本数与生成记录不匹配")
    fig = plt.figure(figsize=(11.6, 2.1 + .83 * len(rows)), facecolor=PAPER)
    grid = fig.add_gridspec(len(rows), 4, left=.095, right=.97, bottom=.09,
                            top=.73, wspace=.53, hspace=.36, width_ratios=[1, 2.8, 1.3, 1.5])
    description = "随仓库提供的视听样本" if sample_type == "训练样本" else sample_type
    heading(fig, "把图像和音频交给同一个语言模型", f"{description}    ·    重载权重后逐 token 生成    ·    seed {seed}")
    for i, (row, pred) in enumerate(zip(rows, generated)):
        if row["question"] != pred["question"]:
            raise ValueError(f"第 {i + 1} 条样本的问题与生成记录不匹配")
        panels = [fig.add_subplot(grid[i, j]) for j in range(4)]
        if i == 0:
            for ax, title in zip(panels, ["图像 / 4 × 4", "音频 / 采样序列", "参考回答", "模型输出"]):
                ax.set_title(title, size=12, pad=8)
        pixels = row["image"]
        panels[0].imshow([pixels[n:n+4] for n in range(0, 16, 4)], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        panels[0].text(-.35, .5, f"{i + 1:02}", transform=panels[0].transAxes, ha="right", va="center", color=MUTED)
        panels[1].axhline(0, color=GRID, lw=1)
        panels[1].plot(row["audio"], color=TEAL, lw=2, marker="o", ms=3)
        panels[1].set_ylim(-1.18, 1.18)
        for ax, answer, color in zip(panels[2:], [row["answer"], pred["answer"]], [INK, ORANGE]):
            ax.text(.5, .5, answer, transform=ax.transAxes, ha="center", va="center", size=15,
                    fontfamily=["DejaVu Sans", plt.rcParams["font.family"][0]], color=color)
        for ax in panels:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        for ax in panels[2:]:
            ax.set_facecolor(PAPER)
    correct = sum(row["answer"] == pred["answer"] for row, pred in zip(rows, generated))
    fig.text(.045, .034, f'问题“{rows[0]["question"]}”    ·    完全匹配 {correct}/{len(rows)} 条{sample_type}', color=MUTED)
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def draw_comparison(info, output):
    split = {"test": "留出测试", "dev": "开发"}[info["split"]]
    fig, panels = plt.subplots(1, 2, figsize=(11.6, 4.8), facecolor=PAPER)
    fig.subplots_adjust(left=.075, right=.975, top=.67, bottom=.20, wspace=.23)
    heading(fig, "同一批新样本，三个训练条件", f'{split}集    ·    {info["backend"]}    ·    seed {info["seed"]}')
    cases = [
        ("决策 / 颜色与新名词", info["decision"], ["baseline", "last", "mean"], ["第一课", "成对数据\n末位池化", "成对数据\n均值池化"]),
        ("视听 / 原始文本完全匹配", info["audiovisual"], ["baseline", "prefix", "readout"], ["第一课", "扩充训练\n仅前缀", "扩充训练\n交互读出"]),
    ]
    for ax, (title, scores, keys, labels) in zip(panels, cases):
        records = [scores[key] for key in keys]
        values = [r["correct"] / r["n"] for r in records]
        ax.bar(range(3), values, width=.57, color=[ORANGE, "#78929d", TEAL])
        for i, (v, r) in enumerate(zip(values, records)):
            ax.text(i, v+.035, f'{r["correct"]}/{r["n"]}', ha="center", color=INK)
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_ylim(0, 1.21)
        ax.set_xticks(range(3), labels)
        ax.set_yticks([0, .5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", color=GRID)
    fig.savefig(output / "heldout-comparison.png", dpi=160)
    plt.close(fig)
    rows = info["example_quartet"]
    outputs = info["audiovisual"]["readout"]["predictions"][:len(rows)]
    generated = [dict(question=r["question"], answer=p["raw"]) for r,p in zip(rows, outputs)]
    draw_examples(rows, generated, info["seed"], output/"heldout-audiovisual.png", f"{split}样本")


def draw_bilingual(report, output):
    run = report["runs"]["pretrain"]
    train = run["training"]
    fig, (loss, tasks) = plt.subplots(1, 2, figsize=(12, 5), facecolor=PAPER)
    fig.subplots_adjust(left=.08, right=.95, top=.70, bottom=.22, wspace=.35)
    heading(fig, "125M 双语训练与留出问答",
            f'M2 Pro · Metal    /    预训练 {train["train_targets_per_epoch"]:,} 个有效 token    /    40 道固定问题')
    samples = run["logged_batches"]
    loss.plot([b["step"] for b in samples], [b["nll"] for b in samples],
              marker="o", ms=4, lw=2.3, color=TEAL, label="训练批 NLL（每 8 步采样）")
    val = train["validation"]
    loss.scatter([samples[-1]["step"]], [val["nll"]], marker="D", s=52,
                 color=ORANGE, label=f'验证 NLL {val["nll"]:.2f}（{val["tokens"]} token）', zorder=4)
    loss.set_title("预训练期间的负对数似然", loc="left")
    loss.set_xlabel("AdamW 更新步数")
    loss.set_ylabel("NLL")
    loss.legend(loc="upper right", fontsize=9, frameon=False)
    loss.grid(axis="y", color=GRID)
    keys = ["language", "en-zh", "zh-en", "transfer"]
    titles = ["语种识别", "英译中", "中译英", "跨任务"]
    for stage, name, shift, color in [("pretrain","预训练",-.16,MUTED), ("sft","SFT",.16,TEAL)]:
        scores = [report["results"][stage][k] for k in keys]
        rates = [s["exact"]/s["n"] for s in scores]
        y = [i + shift for i in range(len(keys))]
        tasks.scatter(rates, y, s=48, color=color, label=name)
        for value, pos, score in zip(rates, y, scores):
            tasks.annotate(f'{score["exact"]}/{score["n"]}', (value, pos),
                           xytext=(8,0), textcoords="offset points", va="center", fontsize=9, color=color)
    tasks.set_title("固定问题的完全匹配率", loc="left")
    tasks.set_yticks(range(4), titles)
    tasks.set_ylim(3.5,-.6)
    tasks.set_xlim(-.06,1.18)
    tasks.set_xticks([0,.25,.5,.75,1])
    tasks.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    tasks.grid(axis="x", color=GRID)
    tasks.legend(loc="lower right", frameon=False)
    fig.text(.08, .065, "两份权重使用相同的聊天模板和贪心解码。翻译只设一个参考答案。", color=MUTED)
    fig.savefig(output / "m2-bilingual-125m.png", dpi=160)
    plt.close(fig)


def draw_bilingual_v2(report, output):
    from matplotlib.colors import LinearSegmentedColormap
    stages = ["old-sft", "legacy-translation-sft", "small-sft", "sft"]
    labels = ["旧版", "旧底座\n仅翻译", "新底座\n旧配方", "新版"]
    results = [report["results"][s] for s in stages]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=PAPER)
    fig.subplots_adjust(left=.145, right=.975, top=.70, bottom=.22, wspace=.58)
    heading(fig, "修复词表并扩大语料后，回答如何变化？",
            "M2 Pro · Metal     /     同一批 56 道新题     /     源句置换另用 32 个开发句对")
    left = [[(r["translations"][kind]["distinct_answers"], r["translations"][kind]["n"]) for r in results]
            for kind in ("en-zh", "zh-en")]
    left.append([(r["source_test"]["matched_lower"], r["source_test"]["n"]) for r in results])
    right = [[(r[k]["exact"], r[k]["n"]) for r in results] for k in ("en-zh", "zh-en", "language", "transfer")]
    cmap = LinearSegmentedColormap.from_list("teal_counts", ["#eef5f5", "#8ec8c1", "#07655e"])
    for ax, data, title, names in zip(axes, [left, right],
            ["生成多样性与源句配对", "留出回答完全匹配"],
            [["英译中不同回答", "中译英不同回答", "偏好正确源句"], ["英译中", "中译英", "语种识别", "跨任务"]]):
        ax.imshow([[v/n for v,n in row] for row in data], vmin=0, vmax=1, cmap=cmap, aspect="auto")
        for i, row in enumerate(data):
            for j, (v,n) in enumerate(row):
                ax.text(j,i,f"{v}/{n}",ha="center",va="center",weight="bold",
                        color="white" if v/n>.65 else INK, fontsize=13)
        ax.set_title(title, loc="left")
        ax.set_xticks(range(len(labels)), labels, fontsize=10)
        ax.set_yticks(range(len(names)), names)
        ax.tick_params(length=0,pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.05, .063, "不同回答数按方向统计。偏好正确源句指同一参考译文在原配源句下的 NLL 低于错配。", color=MUTED)
    fig.savefig(output / "m2-bilingual-v2.png", dpi=160)
    plt.close(fig)


def draw_translation(report, output):
    from matplotlib.colors import LinearSegmentedColormap

    scores = report["scores"]
    stages = ("baseline", "memory", "fixed", "crossed", "polarity", "both", "mixed")
    labels = ("上一版 SFT", "只背 12 句对", "固定搭配", "动作搭配交叉", "肯否成对", "交叉＋肯否成对", "再扩自然句")
    colors = LinearSegmentedColormap.from_list("translation", ["#edf3f3", "#81c8be", "#075c60"])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), facecolor=PAPER)
    fig.subplots_adjust(left=.16, right=.97, top=.71, bottom=.14, wspace=.28)
    fig.text(.045, .92, "125M 中英翻译：训练记忆与新句泛化", fontsize=21, weight="bold", color=INK)
    before, after = (scores[k]["memory"]["meaning"] for k in ("baseline", "memory"))
    fig.text(.045, .84, f"M2 Pro · Metal     12 个训练句对双向重答：{before}/24 → {after}/24     下表各组独立从上一版 SFT 开始训练",
             fontsize=11, color=MUTED)
    groups = ("cube", "drinks", "everyday")
    axes_slots = ("action", "object", "polarity")
    full = [[scores[s][g]["meaning"] for g in groups] for s in stages]
    pairs = [[scores[s]["cube"]["minimal_pairs"][k]["slot_both_correct"] for k in axes_slots] for s in stages]
    for ax, values, denominators, title, columns in [
        (axes[0], full, [32, 8, 8], "整句保留原意", ("受控重组", "真实饮品句", "真实日常句")),
        (axes[1], pairs, [16, 16, 16], "只改一个词义，两侧都跟对", ("换动作", "换对象", "翻转肯否")),
    ]:
        fractions = [[x/n for x,n in zip(row, denominators)] for row in values]
        ax.imshow(fractions, cmap=colors, vmin=0, vmax=1, aspect="auto")
        ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=12)
        ax.set_xticks(range(3), columns)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [""]*len(labels))
        ax.tick_params(length=0, pad=7)
        for y, row in enumerate(values):
            for x, count in enumerate(row):
                ax.text(x, y, f"{count}/{denominators[x]}", ha="center", va="center",
                        color="white" if fractions[y][x] > .73 else INK, fontsize=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.045, .035, "同一批冻结题，双向贪心生成。右图每列 16 对，只检查该列改动的含义。全句得分见左图。",
             color=MUTED, fontsize=10)
    fig.savefig(output / "translation-v3.png", dpi=160)
    plt.close(fig)


def draw_daily(report, output):
    from matplotlib.colors import LinearSegmentedColormap

    scores = report["scores"]
    stages = ("before", "small", "wide", "focused")
    groups = ("cannot", "negative", "quantity", "time", "other")
    left = [[scores[s]["directions"][d]["meaning"] for d in ("en-zh", "zh-en")]
            + [scores[s]["both_directions"]] for s in stages]
    right = [[scores[s]["groups"][g]["detail"] for g in groups] for s in stages]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), facecolor=PAPER,
                             gridspec_kw={"width_ratios": [3, 5]})
    fig.subplots_adjust(left=.145, right=.97, top=.68, bottom=.21, wspace=.18)
    fig.text(.04, .91, "同样训练 1,536 步，真实日常句译对了多少？", fontsize=21, color=INK, weight="bold")
    fig.text(.04, .817, "M2 Pro · Metal · 125M     30 个新句对双向自由生成     三组新增训练共用同一起点", color=MUTED, fontsize=11)
    cmap = LinearSegmentedColormap.from_list("daily_quality", ["#f0f3f2", "#a9d6cc", "#076365"])
    labels = ["训练前", "384 对重放", "3,072 对广泛采样", "3,072 对难点采样"]
    for ax, vals, denom, title, cols in [
        (axes[0], left, 30, "整句原意正确", ("英译中", "中译英", "双向都对")),
        (axes[1], right, 12, "该题指定的关键含义正确", ("不能", "一般否定", "数量", "时间与体", "其他日常")),
    ]:
        ax.imshow([[v/denom for v in row] for row in vals], vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_title(title, loc="left", weight="bold", fontsize=13)
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(4), labels if ax is axes[0] else [""]*4)
        ax.tick_params(length=0, pad=8)
        for y,row in enumerate(vals):
            for x,v in enumerate(row):
                ax.text(x,y,f"{v}/{denom}", ha="center", va="center", color="white" if v/denom>.78 else INK, fontsize=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.04,.045,"每个类别 6 对、12 个翻译方向。右图允许其他细节有误。另有 2 个近似旧句对只作诊断。", fontsize=10, color=MUTED)
    fig.savefig(output / "daily-v4.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="训练输出目录或留出评测 JSON 报告")
    parser.add_argument("output", type=Path, help="保存 PNG 的目录")
    parser.add_argument("--font", type=Path, help="可选的中文字体文件")
    args = parser.parse_args()
    fonts = ["Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"]
    if args.font:
        font_manager.fontManager.addfont(args.font)
        fonts.insert(0, font_manager.FontProperties(fname=args.font).get_name())
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((name for name in fonts if name in available), None)
    if not font:
        parser.error("请用 --font 指定一个包含简体中文的字体文件")
    plt.rcParams.update({"font.family": [font, "DejaVu Sans"], "font.size": 11,
                         "axes.titlepad": 12, "axes.axisbelow": True, "axes.edgecolor": GRID,
                         "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    args.output.mkdir(parents=True, exist_ok=True)
    if args.run.is_file():
        info = read_json(args.run)
        if info.get("experiment") == "daily-v4":
            draw_daily(info, args.output)
        elif "corpus_evidence" in info and "inference" in info:
            draw_translation(info, args.output)
        elif "development_contrasts" in info:
            draw_bilingual_v2(info, args.output)
        elif "logged_batches" in info.get("runs", {}).get("pretrain", {}):
            draw_bilingual(info, args.output)
        else:
            draw_comparison(info, args.output)
        return
    info = read_json(args.run / "run.json")
    stages = {stage["name"]: stage for stage in info["stages"]}
    draw_training(args.run, args.output, info, stages)
    draw_multimodal(args.run, args.output, info, stages)


if __name__ == "__main__":
    main()
