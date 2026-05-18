#!/usr/bin/env python3
"""
Compute hypothesis-support metrics (H1–H4) from the results database and
log files, and write the corresponding LaTeX table files.

H1 (Modularity / Wiring Ablation)
    Source: raw_runs.csv — rows for backbones mnistcfc / mnist-random-sparse / mnist-dense-cfc
    Output: LTC_CFC_ContinualLearning/tables/wiring_ablation.tex

H2 (Temporal Stability) + H3 (Gradient Isolation)
    Source: paper_results/logs/ — parses tau bimodality, representational
            stability, and gradient cosine similarity from advanced-metrics logs
    Output: LTC_CFC_ContinualLearning/tables/internal_dynamics.tex

H4 (Expressivity — Joint upper bound comparison)
    Source: raw_runs.csv — Joint rows for mnistcfc vs mnistmlp
    Output: printed to stdout and annotated inside wiring_ablation.tex caption

Usage:
    python scripts/analysis/compute_hypothesis_metrics.py \
        [--raw paper_results/results_db/raw_runs.csv] \
        [--logs paper_results/logs] \
        [--out LTC_CFC_ContinualLearning/tables]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="paper_results/results_db/raw_runs.csv",
                   help="Path to raw_runs.csv from build_results_db.py")
    p.add_argument("--logs", default="paper_results/logs",
                   help="Directory containing per-run .log files")
    p.add_argument("--out", default="LTC_CFC_ContinualLearning/tables",
                   help="Directory to write LaTeX .tex files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell(values: np.ndarray, bold: bool = False, sig: str = "") -> str:
    """Format a numpy array as a LaTeX mean±std cell."""
    if len(values) == 0:
        return "--"
    mean = float(values.mean())
    if len(values) == 1:
        inner = f"{mean:.2f}"
    else:
        std = float(values.std(ddof=1))
        inner = f"{mean:.2f} \\pm {std:.2f}"
    if bold:
        inner = f"\\mathbf{{{inner}}}"
    return f"${inner}${sig}"


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 3 or len(a) != len(b):
        return None
    if np.allclose(a, b):
        return None
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return None


def _sig(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p < 0.01:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    return ""


def _aligned_values(df: pd.DataFrame, backbone: str, model: str,
                    buffer: Optional[int]) -> np.ndarray:
    mask = (df["backbone"] == backbone) & (df["model"] == model)
    if buffer is None:
        mask &= df["buffer_size"].isna()
    else:
        mask &= (df["buffer_size"] == buffer)
    return df[mask]["class_il"].dropna().to_numpy()


# ---------------------------------------------------------------------------
# H1 — Wiring Ablation table
# ---------------------------------------------------------------------------

WIRING_BACKBONES: Dict[str, str] = {
    "AutoNCP (Ours)": "mnistcfc",
    "Random-Sparse": "mnist-random-sparse",
    "Dense-CfC": "mnist-dense-cfc",
}

WIRING_ROWS: List[Tuple[str, str, Optional[int]]] = [
    ("SGD",    "sgd",    None),
    ("Joint",  "joint",  None),
    ("ER",     "er",     200),
    ("ER",     "er",     500),
    ("DER++",  "derpp",  200),
    ("DER++",  "derpp",  500),
    ("ER-ACE", "er_ace", 200),
]


def build_wiring_table(df: pd.DataFrame, out_dir: Path) -> None:
    """Write wiring_ablation.tex — H1 evidence."""
    available = {name: key for name, key in WIRING_BACKBONES.items()
                 if key in df["backbone"].unique()}

    if not available:
        print("[H1] No wiring-ablation backbone data found in raw_runs.csv. "
              "Run `make run-wiring-ablation` first.", file=sys.stderr)
        _write_placeholder_wiring(out_dir)
        return

    headers = list(available.keys())
    col_spec = "l" + "l" + "c" * len(headers)

    lines: List[str] = []
    for label, model, buffer in WIRING_ROWS:
        row_vals: Dict[str, np.ndarray] = {
            name: _aligned_values(df, key, model, buffer)
            for name, key in available.items()
        }
        if all(len(v) == 0 for v in row_vals.values()):
            continue
        # AutoNCP is the proposed; compare to random-sparse baseline
        ancp_vals = row_vals.get("AutoNCP (Ours)", np.array([]))
        rs_vals   = row_vals.get("Random-Sparse",  np.array([]))
        p = _wilcoxon_p(ancp_vals, rs_vals)

        buf_label = str(buffer) if buffer else "--"
        cells = []
        for name in headers:
            v = row_vals.get(name, np.array([]))
            is_best = (name == "AutoNCP (Ours)" and len(ancp_vals) > 0 and
                       (len(rs_vals) == 0 or ancp_vals.mean() >= rs_vals.mean()))
            sig_str = _sig(p) if name == "AutoNCP (Ours)" else ""
            cells.append(_cell(v, bold=is_best, sig=sig_str))

        lines.append(f"  {label} & {buf_label} & " + " & ".join(cells) + r" \\")

    header_row = "\\textbf{Method} & \\textbf{Buffer} & " + \
                 " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\"

    n_seeds = int(df[df["backbone"] == list(available.values())[0]]["seed"].nunique())
    n_note = f"$n={n_seeds}$ seeds"

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Wiring ablation (H1).} Class-IL accuracy (\%) on Split-MNIST"
        r" comparing structured AutoNCP wiring against control wirings at matched"
        r" parameter count. $^*$ $p<0.05$, $^{**}$ $p<0.01$ (paired Wilcoxon,"
        f" {n_note}).}}",
        r"\label{tab:wiring_ablation}",
        r"\resizebox{\columnwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_row,
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    out_path = out_dir / "wiring_ablation.tex"
    out_path.write_text("\n".join(tex) + "\n")
    print(f"[H1] Wrote {out_path}  ({len(lines)} rows, backbones: {list(available.keys())})")


def _write_placeholder_wiring(out_dir: Path) -> None:
    out_path = out_dir / "wiring_ablation.tex"
    if out_path.exists():
        return  # don't overwrite a real table
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Wiring ablation (H1) — pending.} Runs not yet completed.}",
        r"\label{tab:wiring_ablation}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Buffer} & \textbf{Random-Sparse} & \textbf{AutoNCP (Ours)} \\",
        r"\midrule",
        r"-- & -- & -- & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(tex) + "\n")
    print(f"[H1] Wrote placeholder {out_path}")


# ---------------------------------------------------------------------------
# H2/H3 — Internal dynamics from log files
# ---------------------------------------------------------------------------

# Regex patterns for advanced-metrics log lines emitted by Mammoth's
# --enable_advanced_metrics / --enable_tau_monitor instrumentation.
_RE_TAU_BC    = re.compile(r"TAU_BIMODALITY_COEFF\s+([\d.]+)")
_RE_REPSTAB   = re.compile(r"REP_STABILITY\s+([\d.]+)")
_RE_GRADINT   = re.compile(r"GRAD_INTERFERENCE\s+([-\d.]+)")
_RE_SEED      = re.compile(r"seed(\d+)")
_RE_BACKBONE  = re.compile(r"(mnistcfc|cnn-cfc|mnistltc|cnn-ltc)")
_RE_MODEL     = re.compile(r"_(sgd|joint|er|derpp|er_ace|agem)(\d*)")


def _parse_mechanistic_log(path: Path) -> Optional[Dict]:
    """Extract mechanistic metrics from one advanced-metrics log file."""
    text = path.read_text(errors="replace")
    tau_vals  = [float(m.group(1)) for m in _RE_TAU_BC.finditer(text)]
    rep_vals  = [float(m.group(1)) for m in _RE_REPSTAB.finditer(text)]
    grad_vals = [float(m.group(1)) for m in _RE_GRADINT.finditer(text)]

    if not tau_vals and not rep_vals and not grad_vals:
        return None

    name = path.stem
    seed_m = _RE_SEED.search(name)
    bb_m   = _RE_BACKBONE.search(name)
    mod_m  = _RE_MODEL.search(name)

    return {
        "backbone":  bb_m.group(1) if bb_m else "unknown",
        "model":     (mod_m.group(1) + (mod_m.group(2) or "")) if mod_m else "unknown",
        "seed":      int(seed_m.group(1)) if seed_m else -1,
        "tau_bc":    float(np.mean(tau_vals))  if tau_vals  else float("nan"),
        "rep_stab":  float(np.mean(rep_vals))  if rep_vals  else float("nan"),
        "grad_int":  float(np.mean(grad_vals)) if grad_vals else float("nan"),
    }


_MECH_METHODS = [
    ("SGD",       "sgd",    None),
    ("ER (200)",  "er",     200),
    ("ER (500)",  "er",     500),
    ("DER++ (500)", "derpp", 500),
]
_MECH_BACKBONES = ["mnistcfc", "cnn-cfc"]


def build_internal_dynamics_table(log_dir: Path, out_dir: Path) -> None:
    """Parse advanced-metrics logs and write internal_dynamics.tex."""
    log_files = list(log_dir.glob("*.log"))
    records = [r for f in log_files for r in [_parse_mechanistic_log(f)] if r]

    if not records:
        print("[H2/H3] No advanced-metrics log data found. "
              "Run `make run-mechanistic` first.", file=sys.stderr)
        _write_placeholder_dynamics(out_dir)
        return

    df = pd.DataFrame(records)
    out_path = out_dir / "internal_dynamics.tex"
    rows: List[str] = []

    for bb in _MECH_BACKBONES:
        sub = df[df["backbone"] == bb]
        if sub.empty:
            continue
        for label, model, buffer in _MECH_METHODS:
            mask = sub["model"].str.startswith(model)
            if buffer is not None:
                mask &= sub["model"].str.contains(str(buffer))
            sel = sub[mask]
            if sel.empty:
                rows.append(f"  {bb} & {label} & -- & -- & -- \\\\")
                continue
            tau_c = _cell(sel["tau_bc"].dropna().to_numpy())
            rep_c = _cell(sel["rep_stab"].dropna().to_numpy())
            grad_c = _cell(sel["grad_int"].dropna().to_numpy())
            rows.append(f"  {bb} & {label} & {tau_c} & {rep_c} & {grad_c} \\\\")

    n_files = len(records)
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Internal dynamics (H2/H3).} Mechanistic metrics for CfC"
        r" on Split-MNIST and Split-CIFAR-10."
        r" \textbf{Tau BC}: bimodality coefficient ($>$0.555 = bimodal, H2)."
        r" \textbf{RepStab}: cosine similarity of penultimate representations"
        r" across consecutive tasks."
        r" \textbf{GradInt}: mean gradient cosine similarity between current-task"
        r" and replay-buffer batches (near 0 = low interference, H3)."
        f" Averaged over available runs ($n={n_files}$ log files).}}",
        r"\label{tab:internal_dynamics}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Backbone} & \textbf{Method} & \textbf{Tau BC} $\uparrow$"
        r" & \textbf{RepStab} $\uparrow$ & \textbf{GradInt} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{2pt}\\{\footnotesize $>$0.555 on Tau BC indicates a bimodal"
        r" $\tau$ distribution (H2). GradInt near 0 indicates gradient orthogonality (H3).}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(tex) + "\n")
    print(f"[H2/H3] Wrote {out_path}  ({len(rows)} rows from {n_files} log files)")


def _write_placeholder_dynamics(out_dir: Path) -> None:
    out_path = out_dir / "internal_dynamics.tex"
    if out_path.exists():
        return
    tex = [
        r"% PLACEHOLDER — run `make run-mechanistic` then `make hypothesis-metrics`",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Internal dynamics (H2/H3) — pending.} Mechanistic runs not yet complete.}",
        r"\label{tab:internal_dynamics}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Backbone} & \textbf{Method} & \textbf{Tau BC} $\uparrow$"
        r" & \textbf{RepStab} $\uparrow$ & \textbf{GradInt} \\",
        r"\midrule",
        r"-- & -- & -- & -- & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(tex) + "\n")
    print(f"[H2/H3] Wrote placeholder {out_path}")


# ---------------------------------------------------------------------------
# H4 — Joint upper bound comparison (printed to stdout)
# ---------------------------------------------------------------------------

def report_h4(df: pd.DataFrame) -> None:
    """Print H4 (expressivity) evidence: Joint upper bound CfC vs MLP."""
    joint_rows = df[df["model"] == "joint"].copy()
    for ds, pairs in [("mnist", ("mnistcfc", "mnistmlp")),
                      ("cifar10", ("cnn-cfc", "resnet18"))]:
        sub = joint_rows[joint_rows["dataset"] == ds]
        for bb in pairs:
            v = sub[sub["backbone"] == bb]["class_il"].dropna().to_numpy()
            print(f"[H4] {ds:8s}  {bb:18s}  Joint = {_cell(v)}  (n={len(v)})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = Path(args.raw)
    if not raw_path.exists():
        print(f"[ERROR] raw_runs.csv not found at {raw_path}. "
              "Run `make results-db` first.", file=sys.stderr)
        # Still try to build dynamics table from logs
        build_internal_dynamics_table(Path(args.logs), out_dir)
        sys.exit(1)

    df = pd.read_csv(raw_path)
    # Normalise column types
    df["buffer_size"] = pd.to_numeric(df["buffer_size"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["class_il"] = pd.to_numeric(df["class_il"], errors="coerce")

    print("=== H1: Wiring Ablation ===")
    build_wiring_table(df, out_dir)

    print("\n=== H2/H3: Internal Dynamics ===")
    build_internal_dynamics_table(Path(args.logs), out_dir)

    print("\n=== H4: Expressivity (Joint upper bounds) ===")
    report_h4(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
