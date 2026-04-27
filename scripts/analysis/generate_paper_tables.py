#!/usr/bin/env python3
"""
Generate paper-ready LaTeX tables with paired Wilcoxon significance markers.

Reads `paper_results/results_db/raw_runs.csv` (produced by
`build_results_db.py`) and writes:
  LTC_CFC_ContinualLearning/tables/mnist_results.tex
  LTC_CFC_ContinualLearning/tables/cifar_results.tex
  LTC_CFC_ContinualLearning/tables/cnn_cfc_results.tex

For each (dataset, model, buffer_size) cell we run a paired Wilcoxon
signed-rank test across seeds between the proposed backbone and the baseline
backbone, and annotate the proposed value with a single marker:
    *  : p < 0.05
    ** : p < 0.01

Cells with fewer than 3 paired seeds are reported without a marker.

Usage:
    python scripts/analysis/generate_paper_tables.py \
        [--raw paper_results/results_db/raw_runs.csv] \
        [--out LTC_CFC_ContinualLearning/tables]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Backbone pairings: (proposed, baseline) per dataset
BACKBONE_PAIRS: Dict[str, Tuple[str, str]] = {
    "mnist": ("mnistcfc", "mnistmlp"),
    "cifar10": ("cnn-cfc", "resnet18"),
}

# Display names used in the LaTeX header
BACKBONE_DISPLAY: Dict[str, str] = {
    "mnistmlp": "MLP",
    "mnistcfc": "CfC (Ours)",
    "resnet18": "ResNet-18",
    "cnn-cfc": "CNN-CfC (Ours)",
}

# Ordering / labelling of methods inside each table
MNIST_ROWS: List[Dict] = [
    {"section": "Bounds"},
    {"label": "SGD (lower)", "model": "sgd", "buffer": None},
    {"label": "Joint (upper)", "model": "joint", "buffer": None},
    {"section": "Replay-based methods"},
    {"label": "ER", "model": "er", "buffer": 200},
    {"label": "ER", "model": "er", "buffer": 500},
    {"label": "DER++", "model": "derpp", "buffer": 200},
    {"label": "DER++", "model": "derpp", "buffer": 500},
    {"label": "ER-ACE", "model": "er-ace", "buffer": 200},
    {"section": "Regularization / constraint methods"},
    {"label": "A-GEM", "model": "agem", "buffer": 200},
]

CIFAR_ROWS: List[Dict] = [
    {"section": "Bounds"},
    {"label": "SGD (lower)", "model": "sgd", "buffer": None},
    {"label": "Joint (upper)", "model": "joint", "buffer": None},
    {"section": "Experience Replay (ER)"},
    {"label": "ER", "model": "er", "buffer": 200},
    {"label": "ER", "model": "er", "buffer": 500},
    {"label": "ER", "model": "er", "buffer": 1000},
    {"section": "Dark Experience Replay (DER++)"},
    {"label": "DER++", "model": "derpp", "buffer": 200},
    {"label": "DER++", "model": "derpp", "buffer": 500},
    {"label": "DER++", "model": "derpp", "buffer": 1000},
    {"section": "ER-ACE"},
    {"label": "ER-ACE", "model": "er-ace", "buffer": 200},
    {"label": "ER-ACE", "model": "er-ace", "buffer": 500},
    {"label": "ER-ACE", "model": "er-ace", "buffer": 1000},
]


def _wilcoxon(prop: np.ndarray, base: np.ndarray) -> Optional[float]:
    """Paired Wilcoxon. Returns p-value, or None if not computable."""
    if len(prop) != len(base) or len(prop) < 3:
        return None
    if np.allclose(prop, base):
        return None
    try:
        from scipy.stats import wilcoxon  # local import; optional dep
    except ImportError:
        return None
    try:
        # zero_method='wilcox' is default; fall back to a gentle method for small n.
        res = wilcoxon(prop, base, zero_method="wilcox", alternative="two-sided")
        return float(res.pvalue)
    except ValueError:
        return None


def _sig_marker(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p < 0.01:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    return ""


def _cell(values: np.ndarray, sig: str = "", bold: bool = False) -> str:
    if values.size == 0:
        return "--"
    mean = values.mean()
    if values.size == 1:
        inner = f"{mean:.2f}"
    else:
        std = values.std(ddof=1)
        inner = f"{mean:.2f} \\pm {std:.2f}"
    if bold:
        inner = f"\\mathbf{{{inner}}}"
    return f"${inner}${sig}"


def _aligned_seeds(
    df: pd.DataFrame, backbone: str, model: str, buffer: Optional[int]
) -> pd.DataFrame:
    sel = df[(df["backbone"] == backbone) & (df["model"] == model)]
    if buffer is None:
        sel = sel[sel["buffer_size"].isna()]
    else:
        sel = sel[sel["buffer_size"] == buffer]
    return sel.sort_values("seed")[["seed", "class_il"]].dropna()


def _row_for_pair(
    df: pd.DataFrame, prop_bb: str, base_bb: str, model: str, buffer: Optional[int]
) -> Tuple[str, str]:
    """Return (baseline_cell, proposed_cell) LaTeX strings for one row."""
    base = _aligned_seeds(df, base_bb, model, buffer)
    prop = _aligned_seeds(df, prop_bb, model, buffer)

    paired = base.merge(prop, on="seed", suffixes=("_base", "_prop"))
    if paired.empty:
        base_vals = base["class_il"].to_numpy()
        prop_vals = prop["class_il"].to_numpy()
        p = None
    else:
        base_vals = paired["class_il_base"].to_numpy()
        prop_vals = paired["class_il_prop"].to_numpy()
        p = _wilcoxon(prop_vals, base_vals)

    if base_vals.size == 0 and prop_vals.size == 0:
        return "--", "--"

    bold_base = (
        base_vals.size > 0
        and prop_vals.size > 0
        and base_vals.mean() > prop_vals.mean()
    )
    bold_prop = not bold_base and prop_vals.size > 0 and base_vals.size > 0

    return (
        _cell(base_vals, bold=bold_base),
        _cell(prop_vals, sig=_sig_marker(p), bold=bold_prop),
    )


def _render_table(
    df: pd.DataFrame,
    rows: List[Dict],
    *,
    dataset: str,
    proposed_bb: str,
    baseline_bb: str,
    label: str,
    caption: str,
) -> str:
    proposed_name = BACKBONE_DISPLAY.get(proposed_bb, proposed_bb)
    baseline_name = BACKBONE_DISPLAY.get(baseline_bb, baseline_bb)

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{llcc}")
    lines.append("\\toprule")
    lines.append(
        f"\\textbf{{Method}} & \\textbf{{Buffer ($M$)}} & "
        f"\\textbf{{{baseline_name}}} & \\textbf{{{proposed_name}}} \\\\"
    )
    lines.append("\\midrule")

    first_section = True
    for row in rows:
        if "section" in row:
            if not first_section:
                lines.append("\\midrule")
            first_section = False
            lines.append(
                f"\\rowcolor{{gray!10}} \\multicolumn{{4}}{{l}}{{\\textit{{{row['section']}}}}} \\\\"
            )
            continue

        base_cell, prop_cell = _row_for_pair(
            df, proposed_bb, baseline_bb, row["model"], row["buffer"]
        )
        buf_str = "--" if row["buffer"] is None else str(row["buffer"])
        lines.append(
            f"{row['label']} & {buf_str} & {base_cell} & {prop_cell} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\vspace{2pt}\\\\\\footnotesize "
        "Class-IL accuracy (\\%), mean $\\pm$ std over seeds. "
        "$^{*}$ paired Wilcoxon $p<0.05$, "
        "$^{**}$ $p<0.01$ vs.\\ baseline."
    )
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _render_cnn_cfc_breakdown(df: pd.DataFrame) -> str:
    """Standalone CNN-CfC breakdown table (matches existing layout)."""
    rows = [
        ("SGD (Lower Bound)", "sgd", None, 0),
        ("Joint (Upper Bound)", "joint", None, None),
    ]
    methods = [
        ("ER", "er"),
        ("DER++", "derpp"),
        ("ER-ACE", "er-ace"),
    ]
    for buf in (200, 500, 1000):
        for label, model in methods:
            rows.append((label, model, buf, buf))

    joint_vals = _aligned_seeds(df, "cnn-cfc", "joint", None)["class_il"].to_numpy()
    joint_mean = joint_vals.mean() if joint_vals.size else float("nan")

    out: List[str] = []
    out.append("\\begin{table}[h]")
    out.append("\\centering")
    out.append(
        "\\caption{\\textbf{CNN-CfC breakdown on Sequential CIFAR-10.} "
        "Class-IL accuracy (\\%), gap to Joint upper bound, mean $\\pm$ std over seeds.}"
    )
    out.append("\\label{tab:cnn_cfc_results}")
    out.append("\\resizebox{\\linewidth}{!}{")
    out.append("\\begin{tabular}{lccc}")
    out.append("\\toprule")
    out.append(
        "\\textbf{Method} & \\textbf{Buffer ($M$)} & \\textbf{Acc (\\%)} & \\textbf{Gap to Joint} \\\\"
    )
    out.append("\\midrule")
    for label, model, buf, _disp in rows:
        seeds = _aligned_seeds(df, "cnn-cfc", model, buf)["class_il"].to_numpy()
        if seeds.size == 0:
            cell = "--"
            gap = "--"
        else:
            mean = seeds.mean()
            std = seeds.std(ddof=1) if seeds.size > 1 else 0.0
            cell = f"${mean:.2f} \\pm {std:.2f}$"
            gap = (
                f"${mean - joint_mean:+.2f}$"
                if not np.isnan(joint_mean)
                else "--"
            )
        buf_str = "--" if buf is None else ("0" if label.startswith("SGD") else str(buf))
        out.append(f"{label} & {buf_str} & {cell} & {gap} \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def _render_wiring_ablation(df: pd.DataFrame) -> str:
    """Wiring ablation (H1): AutoNCP vs Random-Sparse vs Dense-CfC.

    Reports ER@200 Class-IL accuracy on Split-MNIST and Split-CIFAR-10. Each
    row is one dataset, columns are the three connectivity variants. Cells
    with no runs are emitted as ``--``.
    """
    rows = [
        # (dataset, label, autoncp_bb, random_bb, dense_bb)
        ("mnist", "Split-MNIST", "mnistcfc", "mnist-random-sparse", "mnist-dense-cfc"),
        ("cifar10", "Split-CIFAR-10", "cnn-cfc", "cnn-random-sparse", "cnn-dense-cfc"),
    ]
    model = "er"
    buffer = 200

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{\\textbf{Wiring ablation (H1).} Class-IL accuracy (\\%) on "
        "Split-MNIST and Split-CIFAR-10 with ER ($M{=}200$). Compares "
        "structured AutoNCP wiring against random-sparse and dense-CfC controls "
        "at matched parameter count.}"
    )
    lines.append("\\label{tab:wiring_ablation}")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Dataset} & \\textbf{Random-Sparse} & "
        "\\textbf{Dense-CfC} & \\textbf{AutoNCP (Ours)} \\\\"
    )
    lines.append("\\midrule")
    for dataset, label, autoncp_bb, random_bb, dense_bb in rows:
        sub = df[df["dataset"] == dataset]
        autoncp = _aligned_seeds(sub, autoncp_bb, model, buffer)["class_il"].to_numpy()
        random = _aligned_seeds(sub, random_bb, model, buffer)["class_il"].to_numpy()
        dense = _aligned_seeds(sub, dense_bb, model, buffer)["class_il"].to_numpy()
        means = [random.mean() if random.size else -np.inf,
                 dense.mean() if dense.size else -np.inf,
                 autoncp.mean() if autoncp.size else -np.inf]
        best = int(np.argmax(means)) if max(means) > -np.inf else -1
        cells = [
            _cell(random, bold=(best == 0)),
            _cell(dense, bold=(best == 1)),
            _cell(autoncp, bold=(best == 2)),
        ]
        lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\vspace{2pt}\\\\\\footnotesize "
        "Mean $\\pm$ std over seeds ($n{=}3$). Cells with no completed runs "
        "appear as ``--''."
    )
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="paper_results/results_db/raw_runs.csv")
    ap.add_argument("--out", default="LTC_CFC_ContinualLearning/tables")
    args = ap.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.is_file():
        print(f"[generate_paper_tables] ERROR: {raw_path} not found. "
              f"Run `make results-db` first.")
        return 1

    df = pd.read_csv(raw_path)
    # Normalize backbone / model strings to dashed-lowercase form so runs
    # logged as e.g. mnist_random_sparse vs mnist-random-sparse aggregate.
    for col in ("backbone", "model"):
        df[col] = df[col].astype(str).str.replace("_", "-", regex=False).str.lower()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mnist_tex = _render_table(
        df[df["dataset"] == "mnist"],
        MNIST_ROWS,
        dataset="mnist",
        proposed_bb="mnistcfc",
        baseline_bb="mnistmlp",
        label="tab:mnist_results",
        caption=(
            "\\textbf{Split-MNIST (Class-IL, \\%).} "
            "MLP baseline vs.\\ CfC backbone across CL methods. "
            "Bold marks the better backbone per row."
        ),
    )
    (out_dir / "mnist_results.tex").write_text(mnist_tex)
    print(f"[generate_paper_tables] Wrote {out_dir / 'mnist_results.tex'}")

    cifar_tex = _render_table(
        df[df["dataset"] == "cifar10"],
        CIFAR_ROWS,
        dataset="cifar10",
        proposed_bb="cnn-cfc",
        baseline_bb="resnet18",
        label="tab:cifar_results",
        caption=(
            "\\textbf{Split-CIFAR-10 (Class-IL, \\%).} "
            "ResNet-18 baseline vs.\\ CNN-CfC hybrid across CL methods and buffer sizes."
        ),
    )
    (out_dir / "cifar_results.tex").write_text(cifar_tex)
    print(f"[generate_paper_tables] Wrote {out_dir / 'cifar_results.tex'}")

    cnn_tex = _render_cnn_cfc_breakdown(df[df["dataset"] == "cifar10"])
    (out_dir / "cnn_cfc_results.tex").write_text(cnn_tex)
    print(f"[generate_paper_tables] Wrote {out_dir / 'cnn_cfc_results.tex'}")

    wiring_tex = _render_wiring_ablation(df)
    (out_dir / "wiring_ablation.tex").write_text(wiring_tex)
    print(f"[generate_paper_tables] Wrote {out_dir / 'wiring_ablation.tex'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
