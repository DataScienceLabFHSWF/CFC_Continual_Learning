#!/usr/bin/env bash
# =============================================================================
# Mechanistic-metrics runner (H2: Temporal Stability, H3: Gradient Isolation).
#
# Re-runs a small subset of MNIST and CIFAR-10 cells with
#   --enable_advanced_metrics 1
#   --enable_tau_monitor 1
# so that tau-distribution, representational-stability and gradient-interference
# data is logged to WandB.
#
# Subset (12 + 12 = 24 runs):
#   MNIST  : mnistcfc x {sgd, er@200, er@500, derpp@500} x 3 seeds
#   CIFAR10: cnn-cfc  x {sgd, er@200, er@500, derpp@500} x 3 seeds
#
# Logs are written under paper_results/logs/<exp>_metrics_seed<seed>.log so they
# do not collide with the standard benchmark logs.
#
# Usage: scripts/benchmarks/run_mechanistic.sh [--max-parallel N] [--dry-run]
# =============================================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/home/fneubuerger/CFC_Continual_Learning}"
MAMMOTH_DIR="$WORKSPACE/mammoth"
LOG_DIR="$WORKSPACE/paper_results/logs"
WANDB_ENTITY="${WANDB_ENTITY:-fneubuerger}"
WANDB_PROJECT="${WANDB_PROJECT:-mammoth}"
SEEDS=(0 1 2)
MAX_PARALLEL=4
DRY_RUN=false
TAU_LOG_INTERVAL=200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=true; shift ;;
    --tau-log-interval) TAU_LOG_INTERVAL="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
LOG_DIR="$(realpath "$LOG_DIR")"

is_done() {
  local lf="$1"
  [[ -f "$lf" ]] && grep -q "wandb: Synced\|Run history:" "$lf" 2>/dev/null
}

wait_for_slot() {
  while true; do
    local n
    n=$(tmux list-sessions 2>/dev/null | grep -c '^mech_' || true)
    [[ -z "$n" ]] && n=0
    (( n < MAX_PARALLEL )) && return
    sleep 10
  done
}

launch() {
  local exp="$1"; local seed="$2"; shift 2
  local lf="$LOG_DIR/${exp}_metrics_seed${seed}.log"
  if is_done "$lf"; then
    echo "  skip $exp seed=$seed (done)"; return
  fi
  if $DRY_RUN; then
    echo "[DRY] $exp seed=$seed :: $*"
    return
  fi
  wait_for_slot
  local sess="mech_${exp}_s${seed}"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" "
    cd '$MAMMOTH_DIR'
    source '$WORKSPACE/.venv/bin/activate'
    export WANDB_API_KEY=\$(python3 -c 'import json;print(json.load(open(\"$WORKSPACE/.secrets.json\"))[\"wandb_api_key\"])')
    python utils/main.py $* \
      --seed $seed --num_workers 4 \
      --enable_advanced_metrics 1 --enable_tau_monitor 1 \
      --tau_log_interval $TAU_LOG_INTERVAL \
      --wandb_entity '$WANDB_ENTITY' --wandb_project '$WANDB_PROJECT' \
      --wandb_name '${exp}_metrics_seed${seed}' \
      2>&1 | tee '$lf'
  "
  echo "  start $sess (log: $lf)"
}

# MNIST CfC subset (10 epochs)
for seed in "${SEEDS[@]}"; do
  launch "mnist_cfc_sgd"      "$seed" --dataset seq-mnist --model sgd   --backbone mnistcfc --n_epochs 10 --lr 0.03 --batch_size 32
  launch "mnist_cfc_er200"    "$seed" --dataset seq-mnist --model er    --backbone mnistcfc --n_epochs 10 --lr 0.03 --batch_size 32 --buffer_size 200
  launch "mnist_cfc_er500"    "$seed" --dataset seq-mnist --model er    --backbone mnistcfc --n_epochs 10 --lr 0.03 --batch_size 32 --buffer_size 500
  launch "mnist_cfc_derpp500" "$seed" --dataset seq-mnist --model derpp --backbone mnistcfc --n_epochs 10 --lr 0.03 --batch_size 32 --buffer_size 500 --alpha 0.1 --beta 0.5
done

# CIFAR-10 CNN-CfC subset (50 epochs)
for seed in "${SEEDS[@]}"; do
  launch "cifar_cfc_sgd"      "$seed" --dataset seq-cifar10 --model sgd   --backbone cnn-cfc --n_epochs 50 --lr 0.03 --batch_size 32
  launch "cifar_cfc_er200"    "$seed" --dataset seq-cifar10 --model er    --backbone cnn-cfc --n_epochs 50 --lr 0.03 --batch_size 32 --buffer_size 200
  launch "cifar_cfc_er500"    "$seed" --dataset seq-cifar10 --model er    --backbone cnn-cfc --n_epochs 50 --lr 0.03 --batch_size 32 --buffer_size 500
  launch "cifar_cfc_derpp500" "$seed" --dataset seq-cifar10 --model derpp --backbone cnn-cfc --n_epochs 50 --lr 0.03 --batch_size 32 --buffer_size 500 --alpha 0.1 --beta 0.5
done

echo "Mechanistic-metrics jobs dispatched (max-parallel=$MAX_PARALLEL)."
