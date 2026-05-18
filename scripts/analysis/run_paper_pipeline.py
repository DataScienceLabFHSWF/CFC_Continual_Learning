#!/usr/bin/env python3
"""
Full analysis pipeline orchestrator for the CfC Continual Learning paper.

Runs the following steps in order:
  1. build_results_db.py       — pull WandB runs → raw_runs.csv
  2. generate_paper_tables.py  — raw_runs.csv → LaTeX benchmark tables
  3. compute_hypothesis_metrics.py — raw_runs.csv + logs → wiring/dynamics tables
  4. generate_paper_plots.py   — raw_runs.csv → PNG figures

Each step is a subprocess call so failures are isolated and reported clearly.
Skip steps with --skip-db, --skip-tables, --skip-metrics, --skip-plots.

Usage:
    python scripts/analysis/run_paper_pipeline.py
    python scripts/analysis/run_paper_pipeline.py --skip-db  # reuse existing CSV
    python scripts/analysis/run_paper_pipeline.py --only-tables
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw",    default="paper_results/results_db/raw_runs.csv")
    p.add_argument("--logs",   default="paper_results/logs")
    p.add_argument("--tables", default="LTC_CFC_ContinualLearning/tables")
    p.add_argument("--figs",   default="LTC_CFC_ContinualLearning/figures")
    p.add_argument("--db-out", default="paper_results/results_db")
    p.add_argument("--entity", default=None, help="WandB entity override")
    p.add_argument("--project", default=None, help="WandB project override")
    p.add_argument("--min-year", type=int, default=2025)

    g = p.add_argument_group("skip flags")
    g.add_argument("--skip-db",      action="store_true", help="Skip step 1 (WandB pull)")
    g.add_argument("--skip-tables",  action="store_true", help="Skip step 2 (benchmark tables)")
    g.add_argument("--skip-metrics", action="store_true", help="Skip step 3 (hypothesis metrics)")
    g.add_argument("--skip-plots",   action="store_true", help="Skip step 4 (plots)")
    g.add_argument("--only-tables",  action="store_true",
                   help="Equivalent to --skip-db --skip-metrics --skip-plots")
    return p.parse_args()


def _run(label: str, cmd: list[str]) -> bool:
    """Run a subprocess command, stream output, return True on success."""
    print(f"\n{'='*60}")
    print(f"STEP: {label}")
    print(f"CMD:  {' '.join(cmd)}")
    print("="*60)
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        print(f"\n[FAIL] Step '{label}' exited with code {result.returncode}",
              file=sys.stderr)
        return False
    return True


def main() -> None:
    args = _parse_args()

    if args.only_tables:
        args.skip_db      = True
        args.skip_metrics = True
        args.skip_plots   = True

    failures: list[str] = []
    python = sys.executable

    # Step 1 — WandB pull
    if not args.skip_db:
        cmd = [python, str(HERE / "build_results_db.py"),
               "--output-dir", args.db_out,
               "--min-year", str(args.min_year)]
        if args.entity:
            cmd += ["--entity", args.entity]
        if args.project:
            cmd += ["--project", args.project]
        if not _run("Pull WandB runs → raw_runs.csv", cmd):
            failures.append("build_results_db")

    # Step 2 — Benchmark tables
    if not args.skip_tables:
        cmd = [python, str(HERE / "generate_paper_tables.py"),
               "--raw", args.raw,
               "--out", args.tables]
        if not _run("Generate benchmark LaTeX tables", cmd):
            failures.append("generate_paper_tables")

    # Step 3 — Hypothesis metrics
    if not args.skip_metrics:
        cmd = [python, str(HERE / "compute_hypothesis_metrics.py"),
               "--raw",  args.raw,
               "--logs", args.logs,
               "--out",  args.tables]
        if not _run("Compute hypothesis metrics (H1–H4)", cmd):
            failures.append("compute_hypothesis_metrics")

    # Step 4 — Plots
    if not args.skip_plots:
        cmd = [python, str(HERE / "generate_paper_plots.py"),
               "--raw", args.raw,
               "--out", args.figs]
        if not _run("Generate paper figures", cmd):
            failures.append("generate_paper_plots")

    # Summary
    print(f"\n{'='*60}")
    if failures:
        print(f"Pipeline finished with {len(failures)} failure(s): {failures}")
        sys.exit(1)
    else:
        print("Pipeline finished successfully. All steps passed.")


if __name__ == "__main__":
    main()
