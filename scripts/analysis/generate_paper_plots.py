#!/usr/bin/env python3
"""
Generate publication-quality comparison plots for the paper.

Reads `paper_results/results_db/raw_runs.csv` (produced by build_results_db.py)
and writes PNG figures to LTC_CFC_ContinualLearning/figures/.

Plots generated (one per dataset):
  mnist_baseline_vs_cfc.png  — dumbbell plot, MLP vs CfC on Split-MNIST
  cifar_baseline_vs_cfc.png  — dumbbell plot, ResNet-18 vs CNN-CfC on Split-CIFAR-10

Each row is one (method, buffer) cell: a connecting line from the baseline mean
to the proposed-backbone mean (colored green if the proposed backbone wins,
red otherwise), individual per-seed values as jittered dots, and the delta in
percentage points annotated at the right margin. This replaces the previous
grouped-bar + separate delta-bar pair: it shows the same effect-size
information as the delta plot while also exposing per-seed spread, which a
bar+errorbar chart hides.

Usage:
    python scripts/analysis/generate_paper_plots.py \
        [--raw paper_results/results_db/raw_runs.csv] \
        [--out LTC_CFC_ContinualLearning/figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="paper_results/results_db/raw_runs.csv",
                   help="Path to raw_runs.csv from build_results_db.py")
    p.add_argument("--out", default="LTC_CFC_ContinualLearning/figures",
                   help="Output directory for PNG figures")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Method ordering for display
# ---------------------------------------------------------------------------

# (display_label, model_key, buffer)
MNIST_METHODS: List[Tuple[str, str, Optional[int]]] = [
    ("ER-200",   "er",     200),
    ("ER-500",   "er",     500),
    ("DER++-200","derpp",  200),
    ("DER++-500","derpp",  500),
    ("ER-ACE-200","er_ace",200),
    ("A-GEM-200","agem",   200),
]

CIFAR_METHODS: List[Tuple[str, str, Optional[int]]] = [
    ("ER-200",    "er",     200),
    ("ER-500",    "er",     500),
    ("ER-1000",   "er",     1000),
    ("DER++-200", "derpp",  200),
    ("DER++-500", "derpp",  500),
    ("DER++-1000","derpp",  1000),
    ("ER-ACE-200","er_ace", 200),
    ("ER-ACE-500","er_ace", 500),
    ("ER-ACE-1000","er_ace",1000),
]

COLORS = {"baseline": "#4C78A8", "proposed": "#F58518"}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _seeds(df: pd.DataFrame, dataset: str, backbone: str,
           model: str, buffer: Optional[int]) -> np.ndarray:
    """Return the raw per-seed class_il values for one (backbone, model, buffer) cell."""
    mask = (df["dataset"] == dataset) & (df["backbone"] == backbone) & (df["model"] == model)
    if buffer is None:
        mask &= df["buffer_size"].isna()
    else:
        mask &= (df["buffer_size"] == buffer)
    return df[mask]["class_il"].dropna().to_numpy()


# ---------------------------------------------------------------------------
# Plot: dumbbell (baseline -> proposed mean) with per-seed jitter
# ---------------------------------------------------------------------------

def plot_dumbbell(
    df: pd.DataFrame,
    dataset: str,
    baseline_bb: str,
    proposed_bb: str,
    methods: List[Tuple[str, str, Optional[int]]],
    title: str,
    baseline_label: str,
    proposed_label: str,
    out_path: Path,
    dpi: int,
) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for display, model, buf in methods:
        base_vals = _seeds(df, dataset, baseline_bb, model, buf)
        prop_vals = _seeds(df, dataset, proposed_bb, model, buf)
        if base_vals.size == 0 and prop_vals.size == 0:
            continue
        base_mean = float(base_vals.mean()) if base_vals.size else float("nan")
        prop_mean = float(prop_vals.mean()) if prop_vals.size else float("nan")
        delta = (prop_mean - base_mean
                 if base_vals.size and prop_vals.size else float("nan"))
        rows.append(dict(label=display, base_vals=base_vals, prop_vals=prop_vals,
                          base_mean=base_mean, prop_mean=prop_mean, delta=delta))

    if not rows:
        print(f"  [skip] no data for {dataset} {baseline_bb} vs {proposed_bb}")
        return

    # Order rows by effect size (largest CfC advantage at top) so the plot
    # doubles as a ranked summary, not just a lookup table.
    rows.sort(key=lambda r: (r["delta"] if not np.isnan(r["delta"]) else -np.inf))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.2, max(3.2, n * 0.68)))
    y = np.arange(n)

    all_vals = np.concatenate(
        [r["base_vals"] for r in rows] + [r["prop_vals"] for r in rows]
        + [np.array([0.0, 100.0])]
    )
    x_max = min(100.0, all_vals.max()) + 14.0  # headroom for the delta annotation

    for i, r in enumerate(rows):
        line_color = "#9CA3AF"
        if not np.isnan(r["delta"]):
            line_color = "#2E8B57" if r["delta"] >= 0 else "#C44E52"
        if not (np.isnan(r["base_mean"]) or np.isnan(r["prop_mean"])):
            ax.plot([r["base_mean"], r["prop_mean"]], [i, i],
                    color=line_color, linewidth=2.4, zorder=1, solid_capstyle="round")

        jitter = rng.uniform(-0.14, 0.14, size=r["base_vals"].size)
        ax.scatter(r["base_vals"], np.full_like(r["base_vals"], i, dtype=float) + jitter,
                   color=COLORS["baseline"], s=26, alpha=0.5, zorder=2, edgecolor="none")
        jitter = rng.uniform(-0.14, 0.14, size=r["prop_vals"].size)
        ax.scatter(r["prop_vals"], np.full_like(r["prop_vals"], i, dtype=float) + jitter,
                   color=COLORS["proposed"], s=26, alpha=0.5, zorder=2, edgecolor="none")

        if not np.isnan(r["base_mean"]):
            ax.scatter([r["base_mean"]], [i], color=COLORS["baseline"], s=85,
                       zorder=3, edgecolor="white", linewidth=0.9)
        if not np.isnan(r["prop_mean"]):
            ax.scatter([r["prop_mean"]], [i], color=COLORS["proposed"], s=85,
                       zorder=3, edgecolor="white", linewidth=0.9)

        if not np.isnan(r["delta"]):
            x_right = max(r["base_mean"], r["prop_mean"])
            arrow = "\u2191" if r["delta"] >= 0 else "\u2193"
            ax.text(x_right + 2.0, i, f"{arrow} {abs(r['delta']):.1f} pp", va="center",
                    fontsize=9, color=line_color, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9.5)
    ax.set_xlabel("Class-IL Accuracy (%)", fontsize=10.5)
    ax.set_title(title, fontsize=11.5, pad=10)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.7, n - 0.3)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["baseline"],
                   markersize=9, label=f"{baseline_label} — mean"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["proposed"],
                   markersize=9, label=f"{proposed_label} — mean"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#9CA3AF",
                   markersize=6, alpha=0.6, label="individual seed"),
        plt.Line2D([0], [0], color="#2E8B57", linewidth=2.4, label="CfC/CNN-CfC higher"),
        plt.Line2D([0], [0], color="#C44E52", linewidth=2.4, label="baseline higher"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               frameon=False, fontsize=8.5, ncol=5, columnspacing=1.2)

    fig.text(
        0.01, 0.005,
        "How to read: each row connects the baseline mean to the proposed-backbone mean for one "
        "(method, buffer) cell, sorted by effect size; faint dots are individual seeds; "
        "the arrow gives the signed accuracy delta in percentage points.",
        fontsize=7.5, color="#444444", ha="left", va="bottom", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    raw_path = Path(args.raw)
    out_dir = Path(args.out)

    if not raw_path.exists():
        print(f"[ERROR] {raw_path} not found. Run `make results-db` first.", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_path)
    df["buffer_size"] = pd.to_numeric(df["buffer_size"], errors="coerce")
    df["class_il"]    = pd.to_numeric(df["class_il"], errors="coerce")

    print("Generating MNIST plot …")
    plot_dumbbell(
        df, dataset="mnist",
        baseline_bb="mnistmlp", proposed_bb="mnistcfc",
        methods=MNIST_METHODS,
        title="Split-MNIST: MLP vs CfC (per-seed, ranked by effect size)",
        baseline_label="MLP (baseline)",
        proposed_label="CfC (ours)",
        out_path=out_dir / "mnist_baseline_vs_cfc.png",
        dpi=args.dpi,
    )

    print("Generating CIFAR-10 plot …")
    plot_dumbbell(
        df, dataset="cifar10",
        baseline_bb="resnet18", proposed_bb="cnn-cfc",
        methods=CIFAR_METHODS,
        title="Split-CIFAR-10: ResNet-18 vs CNN-CfC (per-seed, ranked by effect size)",
        baseline_label="ResNet-18 (baseline)",
        proposed_label="CNN-CfC (ours)",
        out_path=out_dir / "cifar_baseline_vs_cfc.png",
        dpi=args.dpi,
    )

    print("Done.")


if __name__ == "__main__":
    main()
