"""Generate publication-quality plots from eval_backbone CSV output.

Reads one or more CSV files written by `eval_backbone.py` and renders:
  1. A horizontal bar chart of all scalar metrics for the latest run.
  2. A metric-vs-run line plot for the history CSV (cross-run comparison).
  3. An alignment-uniformity scatter plot (Wang & Isola 2020 diagnostic).

Usage
-----

    # Single-run bar chart
    python -m shape_foundation.scripts.plot_eval_metrics \
        --csv results/checkpoint_final_eval_val.csv \
        --output results/plots/

    # Cross-run history (line plots over time)
    python -m shape_foundation.scripts.plot_eval_metrics \
        --history results/eval_history.csv \
        --output results/plots/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_GROUPS = {
    "reconstruction": [
        ("recon_smoothl1", "SmoothL1 (β=1.0)", "lower is better"),
        ("recon_mse", "MSE", "lower is better"),
        ("recon_r2", "R²", "higher is better"),
    ],
    "contrastive": [
        ("contrastive_top1_acc", "Top-1 Pair Accuracy", "higher is better"),
        ("contrastive_infonce", "InfoNCE Loss", "lower is better"),
        ("contrastive_alignment", "Alignment", "lower is better"),
        ("contrastive_uniformity", "Uniformity", "more negative is better"),
    ],
    "embedding_geometry": [
        ("pairwise_cosine_mean", "Random-pair cosine (mean)", "0 is ideal"),
        ("pairwise_cosine_std", "Random-pair cosine (std)", "descriptive"),
        ("embedding_norm_mean", "Embedding L2 norm (mean)", "descriptive"),
        ("embedding_norm_std", "Embedding L2 norm (std)", "descriptive"),
    ],
}


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(v: str | float | int | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def plot_single_run(row: dict, out_dir: Path) -> None:
    """Bar chart of all numeric metrics for a single evaluation run."""
    fig, axes = plt.subplots(
        len(METRIC_GROUPS), 1,
        figsize=(9, 2.2 * len(METRIC_GROUPS)),
        constrained_layout=True,
    )
    if len(METRIC_GROUPS) == 1:
        axes = [axes]

    for ax, (group_name, metrics) in zip(axes, METRIC_GROUPS.items()):
        names = []
        values = []
        colors = []
        for key, label, direction in metrics:
            v = _to_float(row.get(key))
            if v is None:
                continue
            names.append(f"{label}\n({direction})")
            values.append(v)
            if "lower" in direction or "more negative" in direction:
                colors.append("#ef4444")
            elif "higher" in direction:
                colors.append("#22c55e")
            else:
                colors.append("#6b7280")

        if not values:
            ax.set_visible(False)
            continue

        bars = ax.barh(names, values, color=colors, alpha=0.75, edgecolor="black")
        ax.set_title(group_name.replace("_", " ").title(), fontsize=12, pad=6)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        for bar, v in zip(bars, values):
            width = bar.get_width()
            ha = "left" if width >= 0 else "right"
            offset = 0.01 * (max(values) - min(values) + 1e-8)
            ax.text(
                width + (offset if width >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}",
                va="center", ha=ha,
                fontsize=9, fontweight="bold",
            )

    fig.suptitle(
        f"Shape Foundation Model — Evaluation Metrics\n"
        f"checkpoint: {row.get('checkpoint', '?')}  ·  split: {row.get('split', '?')}",
        fontsize=13, fontweight="bold",
    )

    out_path = out_dir / "metrics_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_history(rows: list[dict], out_dir: Path) -> None:
    """Line plots of each metric across runs (time-ordered by timestamp)."""
    if len(rows) < 2:
        print("History has <2 runs — skipping line plots.")
        return

    rows_sorted = sorted(rows, key=lambda r: r.get("timestamp", ""))
    labels = [r.get("checkpoint", f"run{i}") for i, r in enumerate(rows_sorted)]

    for group_name, metrics in METRIC_GROUPS.items():
        fig, ax = plt.subplots(figsize=(max(7, len(rows_sorted) * 1.2), 4.5),
                               constrained_layout=True)

        plotted = 0
        for key, label, direction in metrics:
            values = [_to_float(r.get(key)) for r in rows_sorted]
            if any(v is None for v in values):
                continue
            ax.plot(range(len(values)), values, marker="o", label=label, linewidth=2)
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{group_name.replace('_', ' ').title()} — across runs",
                     fontsize=12)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.grid(linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        out_path = out_dir / f"history_{group_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_alignment_uniformity(rows: list[dict], out_dir: Path) -> None:
    """Wang & Isola (2020) scatter — higher-right is better."""
    points = []
    for r in rows:
        a = _to_float(r.get("contrastive_alignment"))
        u = _to_float(r.get("contrastive_uniformity"))
        if a is not None and u is not None:
            points.append((a, u, r.get("checkpoint", "")))

    if not points:
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, s=140, c="#2563eb", edgecolors="black", zorder=3)
    for x, y, name in points:
        ax.annotate(name, (x, y), xytext=(8, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold")

    ax.set_xlabel("Alignment  (E‖f(x) − f(y)‖² over positive pairs, lower = better)",
                  fontsize=11)
    ax.set_ylabel("Uniformity  (log E exp(−t‖f(x) − f(y)‖²), lower = better)",
                  fontsize=11)
    ax.set_title("Contrastive Embedding Quality\n(Wang & Isola 2020 diagnostic)",
                 fontsize=12, pad=10)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate(
        "ideal region →\n(low alignment,\n low uniformity)",
        xy=(0.02, 0.05), xycoords="axes fraction",
        fontsize=9, alpha=0.7, style="italic",
    )

    out_path = out_dir / "alignment_uniformity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot eval_backbone metrics")
    parser.add_argument("--csv", type=str, default=None,
                        help="Single-run CSV (from eval_backbone --output)")
    parser.add_argument("--history", type=str, default=None,
                        help="History CSV (from eval_backbone --history)")
    parser.add_argument("--output", type=str, default="results/plots/")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        rows = _read_csv(Path(args.csv))
        if rows:
            plot_single_run(rows[-1], out_dir)
            plot_alignment_uniformity(rows, out_dir)

    if args.history:
        rows = _read_csv(Path(args.history))
        if rows:
            plot_single_run(rows[-1], out_dir)
            plot_history(rows, out_dir)
            plot_alignment_uniformity(rows, out_dir)

    if not args.csv and not args.history:
        parser.error("Provide --csv or --history (or both)")


if __name__ == "__main__":
    main()
