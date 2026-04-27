#!/usr/bin/env python3
"""Check WandB experiment status and identify missing runs."""
import json
import sys
from collections import Counter

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Run: pip install wandb")
    sys.exit(1)

api = wandb.Api()
with open('.secrets.json') as f:
    entity = json.load(f).get('WANDB_ENTITY', 'fneubuerger')

runs = api.runs(f'{entity}/mammoth', filters={'created_at': {'$gt': '2025-01-01'}})

states = Counter()
backbones = Counter()
methods = Counter()

for r in runs:
    states[r.state] += 1
    if r.state == 'finished':
        bb = r.config.get('backbone', '?')
        model = r.config.get('model', '?')
        backbones[bb] += 1
        methods[(bb, model)] += 1

print("=== Run States ===")
for s, c in sorted(states.items()):
    print(f"  {s}: {c}")

print(f"\n=== Finished Runs by Backbone ({sum(backbones.values())} total) ===")
for b, c in sorted(backbones.items()):
    print(f"  {b:25s} {c:3d} runs")

# Check missing ablations
REQUIRED_ABLATIONS = {
    'mnistltc', 'mnist-random-sparse',
    'cnn-ltc', 'cnn-random-sparse',
    'tepltc', 'tep-random-sparse',
}
present = set(backbones.keys())
missing = REQUIRED_ABLATIONS - present
if missing:
    print(f"\n=== Missing Ablation Backbones ===")
    for m in sorted(missing):
        print(f"  {m}")
else:
    print("\nAll ablation backbones have runs!")

# Check TEP status
tep_runs = [r for r in runs if r.state == 'finished' and 'tep' in r.config.get('backbone', '')]
if tep_runs:
    print(f"\n=== TEP Status ({len(tep_runs)} finished runs) ===")
    from datetime import datetime
    old = sum(1 for r in tep_runs if str(r.created_at)[:7] < '2025-12')
    new = len(tep_runs) - old
    print(f"  Pre-fix (broken): {old}")
    print(f"  Post-fix: {new}")
    if old > 0 and new == 0:
        print("  WARNING: All TEP runs are from before the dataset fix!")
        print("  Run: make run-tep")
