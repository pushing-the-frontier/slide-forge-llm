"""Generate IEEE-quality plots from aggregate evaluation results.

Usage:
    python -m training.generate_plots \
        --input outputs/output_all48/aggregate_results.json \
        --output-dir paper/figures
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

MODEL_ORDER = ["finetuned", "base", "claude-opus", "claude-sonnet", "llama4-scout", "gpt-oss-120b"]
COLORS = {
    "finetuned": "#2E86AB",
    "base": "#A23B72",
    "claude-opus": "#F18F01",
    "claude-sonnet": "#E8651A",
    "gpt-oss-120b": "#95A5A6",
    "llama4-scout": "#27AE60",
}
LABELS = {
    "finetuned": "Fine-tuned (Ours)",
    "base": "Base Qwen 7B",
    "claude-opus": "Claude Opus 4.6",
    "claude-sonnet": "Claude Sonnet 4.6",
    "gpt-oss-120b": "GPT OSS 120B",
    "llama4-scout": "Llama 4 Scout",
}
SUB_LABELS = {
    "code_rules": "Code\nRules",
    "render_quality": "Render\nQuality",
    "content_quality": "Content\nQuality",
    "claude_aesthetic_html": "Aesthetic\n(HTML)",
    "claude_aesthetic_visual": "Aesthetic\n(Visual)",
    "brief_reconstruction": "Brief\nRecon.",
}


def _active_models(aggregate):
    return [m for m in MODEL_ORDER if m in aggregate]


def _save(fig, output_dir, name):
    fig.savefig(f"{output_dir}/{name}.png")
    fig.savefig(f"{output_dir}/{name}.pdf")
    plt.close(fig)
    print(f"  Saved {name}.png/pdf")


def plot_radar(aggregate, output_dir):
    """Radar chart comparing sub-scores across all models."""
    models = _active_models(aggregate)
    categories = list(SUB_LABELS.keys())
    cat_labels = list(SUB_LABELS.values())
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))

    for model in models:
        values = [aggregate[model]["sub_scores"][c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.6, markersize=3,
                color=COLORS[model], label=LABELS[model])
        ax.fill(angles, values, alpha=0.05, color=COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), framealpha=0.9, fontsize=7.5)
    ax.set_title("Reward Component Comparison", fontweight="bold", pad=20)

    _save(fig, output_dir, "radar_subscores")


def plot_grouped_bars(aggregate, output_dir):
    """Grouped bar chart of sub-scores + overall quality."""
    models = _active_models(aggregate)
    categories = ["overall"] + list(SUB_LABELS.keys())
    cat_labels = ["Overall\nQuality"] + list(SUB_LABELS.values())

    n_models = len(models)
    x = np.arange(len(categories))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, model in enumerate(models):
        vals = [aggregate[model]["overall_quality"]]
        vals += [aggregate[model]["sub_scores"][c] for c in list(SUB_LABELS.keys())]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=LABELS[model],
                      color=COLORS[model], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0.15:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=5.5, rotation=45)

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.legend(framealpha=0.9, ncol=3, loc="upper center", fontsize=7.5)
    ax.set_title("Quality Scores by Component", fontweight="bold")

    _save(fig, output_dir, "grouped_bars_quality")


def plot_efficiency(aggregate, output_dir):
    """Quality vs inference time scatter."""
    models = [m for m in _active_models(aggregate)
              if aggregate[m].get("avg_time_per_brief_seconds", 0) > 0]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for model in models:
        a = aggregate[model]
        ax.scatter(a["avg_time_per_brief_seconds"], a["overall_quality"],
                   s=140, color=COLORS[model], zorder=5, edgecolors="black", linewidth=0.8)

    placed = []
    for model in models:
        a = aggregate[model]
        tx, ty = a["avg_time_per_brief_seconds"], a["overall_quality"]
        ox, oy = 40, -40
        for px, py in placed:
            if abs(tx - px) < 40 and abs(ty - py) < 0.06:
                oy = 30
                break
        placed.append((tx, ty))
        ax.annotate(f"{LABELS[model]}\nq={a['overall_quality']:.3f}",
                    xy=(tx, ty), xytext=(ox, oy), textcoords="offset points", fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS[model], alpha=0.15))

    ax.set_xlabel("Avg. Inference Time per Brief (seconds)")
    ax.set_ylabel("Overall Quality Score")
    ax.set_title("Quality vs Inference Cost", fontweight="bold")
    max_t = max(aggregate[m]["avg_time_per_brief_seconds"] for m in models)
    ax.set_xlim(0, max_t * 1.3)
    ax.set_ylim(0.15, 0.95)

    _save(fig, output_dir, "efficiency_tradeoff")


def plot_head_to_head_summary(comparisons, aggregate, output_dir):
    """Stacked horizontal bar showing W/T/L for finetuned vs each model."""
    models = _active_models(aggregate)
    matchups = []
    for m in models:
        if m == "finetuned":
            continue
        key = f"finetuned_vs_{m}"
        if key in comparisons:
            matchups.append((key, f"Fine-tuned vs {LABELS[m]}"))

    if not matchups:
        return

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(matchups) + 1.2))

    y_pos = np.arange(len(matchups))
    for i, (key, label) in enumerate(matchups):
        h = comparisons[key]["head_to_head"]
        w, t, l = h["wins"], h["ties"], h["losses"]
        ax.barh(i, w, color="#2E86AB", edgecolor="white", linewidth=0.5, height=0.5)
        ax.barh(i, t, left=w, color="#95A5A6", edgecolor="white", linewidth=0.5, height=0.5)
        ax.barh(i, l, left=w + t, color="#E74C3C", edgecolor="white", linewidth=0.5, height=0.5)

        if w > 2:
            ax.text(w / 2, i, str(w), ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        if t > 2:
            ax.text(w + t / 2, i, str(t), ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        if l > 2:
            ax.text(w + t + l / 2, i, str(l), ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[1] for m in matchups], fontsize=8)
    ax.set_xlabel("Number of Briefs")
    ax.set_title("Head-to-Head: Fine-tuned vs All Models (48 Briefs)", fontweight="bold")
    ax.legend(
        handles=[
            mpatches.Patch(color="#2E86AB", label="Wins"),
            mpatches.Patch(color="#95A5A6", label="Ties"),
            mpatches.Patch(color="#E74C3C", label="Losses"),
        ],
        loc="lower right", fontsize=8, framealpha=0.9,
    )

    _save(fig, output_dir, "head_to_head_summary")


def plot_completion_and_slides(aggregate, output_dir):
    """Grouped bar for operational metrics."""
    models = _active_models(aggregate)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    metrics = [
        ("Completion Rate (%)",
         [aggregate[m]["completed"] / aggregate[m]["total_briefs"] * 100 for m in models]),
        ("Avg Slides Created",
         [aggregate[m]["avg_slides_created"] for m in models]),
        ("Avg Time per Brief (s)",
         [aggregate[m]["avg_time_per_brief_seconds"] for m in models]),
    ]

    for ax, (title, vals) in zip(axes, metrics):
        bars = ax.bar(range(len(models)), vals,
                      color=[COLORS[m] for m in models], edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([LABELS[m] for m in models], fontsize=6.5, rotation=30, ha="right")
        ax.set_title(title, fontsize=10, fontweight="bold")
        for bar, val in zip(bars, vals):
            fmt = f"{val:.0f}" if val > 10 else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    fmt, ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, max(vals) * 1.22 if max(vals) > 0 else 1)

    plt.tight_layout()
    _save(fig, output_dir, "operational_metrics")


def plot_overall_quality_ranking(aggregate, output_dir):
    """Horizontal bar chart ranking all models by overall quality."""
    models = _active_models(aggregate)
    models_sorted = sorted(models, key=lambda m: aggregate[m]["overall_quality"])

    fig, ax = plt.subplots(figsize=(7, 3.5))

    y_pos = np.arange(len(models_sorted))
    vals = [aggregate[m]["overall_quality"] for m in models_sorted]
    colors = [COLORS[m] for m in models_sorted]

    bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.5, height=0.6)

    for bar, val, model in zip(bars, vals, models_sorted):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([LABELS[m] for m in models_sorted], fontsize=9)
    ax.set_xlabel("Overall Quality Score")
    ax.set_title("Model Ranking by Overall Quality", fontweight="bold")
    ax.set_xlim(0, 1.0)

    _save(fig, output_dir, "overall_quality_ranking")


def plot_per_brief_comparison(comparisons, output_dir):
    """Per-brief quality delta: finetuned vs Claude Opus, sorted."""
    key = "finetuned_vs_claude-opus"
    if key not in comparisons:
        return

    per_brief = comparisons[key]["per_brief"]
    per_brief_sorted = sorted(per_brief, key=lambda x: x["delta"])

    topics = [p["topic"][:30] for p in per_brief_sorted]
    deltas = [p["delta"] for p in per_brief_sorted]
    colors = ["#2E86AB" if d >= 0 else "#E74C3C" for d in deltas]

    fig, ax = plt.subplots(figsize=(6, 10))
    y_pos = np.arange(len(topics))
    ax.barh(y_pos, deltas, color=colors, edgecolor="white", linewidth=0.3, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics, fontsize=6.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Quality Delta (Fine-tuned − Claude Opus)")
    ax.set_title("Per-Brief Quality: Fine-tuned vs Claude Opus", fontweight="bold")

    wins = sum(1 for d in deltas if d > 0.01)
    ties = sum(1 for d in deltas if -0.01 <= d <= 0.01)
    losses = sum(1 for d in deltas if d < -0.01)
    ax.text(0.98, 0.02, f"W/T/L: {wins}/{ties}/{losses}",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    _save(fig, output_dir, "per_brief_finetuned_vs_claude")


def main():
    parser = argparse.ArgumentParser(description="Generate IEEE-quality plots")
    parser.add_argument("--input", default="outputs/output_all48/aggregate_results.json")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)

    aggregate = data["aggregate"]
    comparisons = data["comparisons"]

    print("Generating plots...\n")
    plot_overall_quality_ranking(aggregate, args.output_dir)
    plot_radar(aggregate, args.output_dir)
    plot_grouped_bars(aggregate, args.output_dir)
    plot_efficiency(aggregate, args.output_dir)
    plot_head_to_head_summary(comparisons, aggregate, args.output_dir)
    plot_completion_and_slides(aggregate, args.output_dir)
    plot_per_brief_comparison(comparisons, args.output_dir)
    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
