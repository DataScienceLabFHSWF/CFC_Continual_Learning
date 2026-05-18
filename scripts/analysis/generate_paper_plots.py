#!/usr/bin/env python3
"""
Generate publication-quality comparison and delta plots for the paper.

Reads `paper_results/results_db/raw_runs.csv` (produced by build_results_db.py)
and writes PNG figures to LTC_CFC_ContinualLearning/figures/.

Plots generated:
  mnist_baseline_vs_cfc.png       — bar chart, MLP vs CfC on Split-MNIST
  cifar_baseline_vs_cfc.png       — bar chart, ResNet vs CNN-CfC on Split-CIFAR-10
  mnist_delta_cfc_minus_baseline.png  — horizontal delta bars for MNIST
  cifar_delta_cfc_minus_baseline.png  — horizontal delta bars for CIFAR-10

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

def _agg(df: pd.DataFrame, dataset: str, backbone: str,
         model: str, buffer: Optional[int]) -> Tuple[float, float, int]:
    """Return (mean, std, n) for class_il. std=0, n=0 if no data."""
    mask = (df["dataset"] == dataset) & (df["backbone"] == backbone) & (df["model"] == model)
    if buffer is None:
        mask &= df["buffer_size"].isna()
    else:
        mask &= (df["buffer_size"] == buffer)
    vals = df[mask]["class_il"].dropna().to_numpy()
    if len(vals) == 0:
        return float("nan"), 0.0, 0
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, len(vals)


# ---------------------------------------------------------------------------
# Plot: grouped bar (baseline vs proposed)
# ---------------------------------------------------------------------------

def plot_bar_comparison(
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
    labels, base_means, base_stds, prop_means, prop_stds = [], [], [], [], []
    for display, model, buf in methods:
        bm, bs, bn = _agg(df, dataset, baseline_bb, model, buf)
        pm, ps, pn = _agg(df, dataset, proposed_bb, model, buf)
        if np.isnan(bm) and np.isnan(pm):
            continue
        labels.append(display)
        base_means.append(bm if not np.isnan(bm) else 0)
        base_stds.append(bs)
        prop_means.append(pm if not np.isnan(pm) else 0)
        prop_stds.append(ps)

    if not labels:
        print(f"  [skip] no data for {dataset} {baseline_bb} vs {proposed_bb}")
        return

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 4.8))
    bars_b = ax.bar(x - width / 2, base_means, width=width, yerr=base_stds,
                    capsize=3, label=baseline_label, color=COLORS["baseline"], error_kw={"elinewidth": 1})
    bars_p = ax.bar(x + width / 2, prop_means,  width=width, yerr=prop_stds,
                    capsize=3, label=proposed_label, color=COLORS["proposed"], error_kw={"elinewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Class-IL Accuracy (%)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot: horizontal delta bars
# ---------------------------------------------------------------------------

def plot_delta(
    df: pd.DataFrame,
    dataset: str,
    baseline_bb: str,
    proposed_bb: str,
    methods: List[Tuple[str, str, Optional[int]]],
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    labels, deltas = [], []
    for display, model, buf in methods:
        bm, _, _ = _agg(df, dataset, baseline_bb, model, buf)
        pm, _, _ = _agg(df, dataset, proposed_bb, model, buf)
        if np.isnan(bm) or np.isnan(pm):
            continue
        labels.append(display)
        deltas.append(pm - bm)

    if not labels:
        print(f"  [skip] no delta data for {dataset}")
        return

    colors = ["#2E8B57" if d >= 0 else "#C44E52" for d in deltas]
    fig, ax = plt.subplots(figsize=(6.5, max(3, len(labels) * 0.5)))
    ax.barh(labels, deltas, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ Accuracy (Proposed − Baseline, pp)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
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

    print("Generating MNIST plots …")
    plot_bar_comparison(
        df, dataset="mnist",
        baseline_bb="mnistmlp", proposed_bb="mnistcfc",
        methods=MNIST_METHODS,
        title="Split-MNIST: MLP vs CfC by Method & Buffer",
        baseline_label="MLP (baseline)",
        proposed_label="CfC (ours)",
        out_path=out_dir / "mnist_baseline_vs_cfc.png",
        dpi=args.dpi,
    )
    plot_delta(
        df, dataset="mnist",
        baseline_bb="mnistmlp", proposed_bb="mnistcfc",
        methods=MNIST_METHODS,
        title="Split-MNIST: CfC − MLP accuracy delta",
        out_path=out_dir / "mnist_delta_cfc_minus_baseline.png",
        dpi=args.dpi,
    )

    print("Generating CIFAR-10 plots …")
    plot_bar_comparison(
        df, dataset="cifar10",
        baseline_bb="resnet18", proposed_bb="cnn-cfc",
        methods=CIFAR_METHODS,
        title="Split-CIFAR-10: ResNet-18 vs CNN-CfC by Method & Buffer",
        baseline_label="ResNet-18 (baseline)",
        proposed_label="CNN-CfC (ours)",
        out_path=out_dir / "cifar_baseline_vs_cfc.png",
        dpi=args.dpi,
    )
    plot_delta(
        df, dataset="cifar10",
        baseline_bb="resnet18", proposed_bb="cnn-cfc",
        methods=CIFAR_METHODS,
        title="Split-CIFAR-10: CNN-CfC − ResNet-18 accuracy delta",
        out_path=out_dir / "cifar_delta_cfc_minus_baseline.png",
        dpi=args.dpi,
    )

    print("Done.")


if __name__ == "__main__":
    main()
