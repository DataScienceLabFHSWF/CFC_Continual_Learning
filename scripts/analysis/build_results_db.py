#!/usr/bin/env python3
"""
Source-of-truth results database for the CfC Continual Learning paper.

Pulls every finished run from WandB and produces:
- raw_runs.csv: one row per (dataset, backbone, model, buffer_size, seed)
- summary.csv : aggregated mean/std/n by (dataset, backbone, model, buffer_size)

Critically, this script reads `backbone` directly from each run's WandB config
(NOT from the run name), fixing the bug where the previous analysis collapsed
across backbones and produced misleading averages (see G11 in the publication
plan).

Usage:
    python scripts/analysis/build_results_db.py \
        --output-dir paper_results/results_db \
        [--entity fneubuerger] [--project mammoth] [--min-year 2025]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Canonical column order for raw_runs.csv
RAW_COLUMNS: List[str] = [
    "run_id",
    "run_name",
    "state",
    "created_at",
    "dataset",
    "backbone",
    "model",
    "buffer_size",
    "seed",
    "n_epochs",
    "lr",
    "batch_size",
    "class_il",
    "task_il",
    "forgetting",
    "runtime_seconds",
    "wandb_url",
]

# Group keys for the summary
GROUP_KEYS: List[str] = ["dataset", "backbone", "model", "buffer_size"]


def _load_wandb_api():
    """Authenticate against WandB using `.secrets.json` if present, else env."""
    import wandb  # local import so the script can be `--help`'d offline

    secrets = Path(__file__).resolve().parents[2] / ".secrets.json"
    if secrets.is_file():
        with secrets.open() as f:
            data = json.load(f)
        api_key = data.get("wandb_api_key") or data.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=True)
    return wandb.Api(timeout=60)


def _safe_int(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def _normalize_dataset(name: Optional[str]) -> Optional[str]:
    """Map mammoth dataset identifiers to short paper names."""
    if not name:
        return None
    name = str(name).lower()
    if "mnist" in name and "perm" not in name and "rot" not in name:
        return "mnist"
    if "cifar10" in name or "cifar-10" in name:
        return "cifar10"
    if "tennessee" in name or name.startswith("tep"):
        return "tep"
    return name


def _extract_run_row(run) -> Optional[Dict[str, Any]]:
    """Convert a single WandB run into a raw_runs row, or None if unusable."""
    cfg = {k: v for k, v in run.config.items() if not k.startswith("_")}
    summary = dict(run.summary._json_dict)

    dataset = _normalize_dataset(cfg.get("dataset"))
    backbone = cfg.get("backbone")
    if backbone is not None:
        # Mammoth normalizes _<->- internally; align our keys to dashed form so
        # runs launched as e.g. mnist_random_sparse and mnist-random-sparse
        # aggregate into a single cell.
        backbone = str(backbone).replace("_", "-").lower()
    model = cfg.get("model")
    if model is not None:
        model = str(model).replace("_", "-").lower()
    seed = _safe_int(cfg.get("seed"))

    # Skip runs without the four primary keys – they can't be aggregated.
    if not dataset or not backbone or not model or seed is None:
        return None

    class_il = _safe_float(summary.get("RESULT_class_mean_accs"))
    task_il = _safe_float(summary.get("RESULT_task_mean_accs"))
    if class_il is None and task_il is None:
        return None  # crashed / no metrics

    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "dataset": dataset,
        "backbone": backbone,
        "model": model,
        "buffer_size": _safe_int(cfg.get("buffer_size")),
        "seed": seed,
        "n_epochs": _safe_int(cfg.get("n_epochs")),
        "lr": _safe_float(cfg.get("lr")),
        "batch_size": _safe_int(cfg.get("batch_size")),
        "class_il": class_il,
        "task_il": task_il,
        "forgetting": _safe_float(
            summary.get("forgetting") or summary.get("RESULT_forgetting")
        ),
        "runtime_seconds": _safe_float(summary.get("_runtime")),
        "wandb_url": summary.get("wandb_url"),
    }


def fetch_raw(
    entity: str, project: str, min_year: int, state: str = "finished"
) -> pd.DataFrame:
    api = _load_wandb_api()
    print(f"[build_results_db] Querying {entity}/{project} (state={state})...")
    runs = api.runs(f"{entity}/{project}", filters={"state": state})

    rows: List[Dict[str, Any]] = []
    skipped = 0
    too_old = 0
    for run in runs:
        try:
            created = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
            if created.year < min_year:
                too_old += 1
                continue
        except Exception:
            pass
        row = _extract_run_row(run)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    print(
        f"[build_results_db] Collected {len(df)} usable runs "
        f"(skipped {skipped} unusable, {too_old} pre-{min_year})."
    )
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent run per (dataset, backbone, model, buffer, seed)."""
    if df.empty:
        return df
    df = df.copy()
    df["_created"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.sort_values("_created").drop_duplicates(
        subset=GROUP_KEYS + ["seed"], keep="last"
    )
    return df.drop(columns="_created").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(GROUP_KEYS, dropna=False).agg(
        n_seeds=("seed", "nunique"),
        class_il_mean=("class_il", "mean"),
        class_il_std=("class_il", "std"),
        task_il_mean=("task_il", "mean"),
        task_il_std=("task_il", "std"),
        forgetting_mean=("forgetting", "mean"),
        forgetting_std=("forgetting", "std"),
    )
    return grouped.round(3).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "fneubuerger"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "mammoth"))
    ap.add_argument("--min-year", type=int, default=2025)
    ap.add_argument("--output-dir", default="paper_results/results_db")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = fetch_raw(args.entity, args.project, args.min_year)
    raw = deduplicate(raw)
    raw_path = out / "raw_runs.csv"
    raw.to_csv(raw_path, index=False)
    print(f"[build_results_db] Wrote {raw_path} ({len(raw)} rows)")

    summary = summarize(raw)
    summary_path = out / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[build_results_db] Wrote {summary_path} ({len(summary)} cells)")

    if not summary.empty:
        n_low = summary[summary["n_seeds"] < 3]
        if not n_low.empty:
            print(
                f"[build_results_db] WARNING: {len(n_low)} cells have <3 seeds; "
                "see incomplete_cells.csv"
            )
            n_low.to_csv(out / "incomplete_cells.csv", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
