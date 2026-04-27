#!/usr/bin/env bash
# =============================================================================
# Wiring ablation runner (H1: Modularity).
#
# Compares:
#   AutoNCP wiring (mnistcfc, cnn-cfc)  -- already covered by main benchmarks
#   Random sparse wiring (mnist-random-sparse, cnn-random-sparse)
#   Dense CfC wiring     (mnist_dense_cfc, cnn_dense_cfc)
#
# Across {SGD, Joint, ER@200, ER@500, ER@1000 (CIFAR only), DER++@500} x 3 seeds.
# Re-uses run_paper_benchmarks.sh's skip-detection by writing logs in the same
# format ($LOG_DIR/<exp>_seed<seed>.log with `wandb: Synced` marker).
#
# Usage:
#   scripts/benchmarks/run_wiring_ablation.sh [--dataset mnist|cifar|all] \
#       [--max-parallel N] [--dry-run]
# =============================================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/home/fneubuerger/CFC_Continual_Learning}"
MAMMOTH_DIR="$WORKSPACE/mammoth"
LOG_DIR="$WORKSPACE/paper_results/logs"
WANDB_ENTITY="${WANDB_ENTITY:-fneubuerger}"
WANDB_PROJECT="${WANDB_PROJECT:-mammoth}"
SEEDS=(0 1 2)
DATASET="all"
MAX_PARALLEL=4
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)      DATASET="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=true; shift ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
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
    n=$(tmux list-sessions 2>/dev/null | grep -c '^abl_' || true)
    [[ -z "$n" ]] && n=0
    (( n < MAX_PARALLEL )) && return
    sleep 10
  done
}

launch() {
  local exp="$1"; local seed="$2"; shift 2
  local lf="$LOG_DIR/${exp}_seed${seed}.log"
  if is_done "$lf"; then
    echo "  skip $exp seed=$seed (done)"; return
  fi
  if $DRY_RUN; then
    echo "[DRY] $exp seed=$seed :: $*"
    return
  fi
  wait_for_slot
  local sess="abl_${exp}_s${seed}"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" "
    cd '$MAMMOTH_DIR'
    source '$WORKSPACE/.venv/bin/activate'
    export WANDB_API_KEY=\$(python3 -c 'import json;print(json.load(open(\"$WORKSPACE/.secrets.json\"))[\"wandb_api_key\"])')
    python utils/main.py $* \
      --seed $seed --num_workers 4 \
      --enable_advanced_metrics 1 --enable_tau_monitor 1 \
      --tau_log_interval 200 \
      --wandb_entity '$WANDB_ENTITY' --wandb_project '$WANDB_PROJECT' \
      --wandb_name '${exp}_seed${seed}' \
      2>&1 | tee '$lf'
  "
  echo "  start $sess (log: $lf)"
}

run_mnist() {
  local n_epochs=10 lr=0.03 bs=32
  for bb in mnist-random-sparse mnist_dense_cfc; do
    local short=${bb//[^a-zA-Z0-9]/}
    for seed in "${SEEDS[@]}"; do
      launch "mnist_${short}_sgd"     "$seed" --dataset seq-mnist --model sgd     --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs
      launch "mnist_${short}_joint"   "$seed" --dataset seq-mnist --model joint   --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs
      launch "mnist_${short}_er200"   "$seed" --dataset seq-mnist --model er      --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs --buffer_size 200
      launch "mnist_${short}_er500"   "$seed" --dataset seq-mnist --model er      --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs --buffer_size 500
      launch "mnist_${short}_derpp500" "$seed" --dataset seq-mnist --model derpp  --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs --buffer_size 500 --alpha 0.1 --beta 0.5
    done
  done
}

run_cifar() {
  local n_epochs=50 lr=0.03 bs=32
  for bb in cnn-random-sparse cnn_dense_cfc; do
    local short=${bb//[^a-zA-Z0-9]/}
    for seed in "${SEEDS[@]}"; do
      launch "cifar_${short}_sgd"     "$seed" --dataset seq-cifar10 --model sgd   --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs
      launch "cifar_${short}_joint"   "$seed" --dataset seq-cifar10 --model joint --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs
      launch "cifar_${short}_er200"   "$seed" --dataset seq-cifar10 --model er    --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs --buffer_size 200
      launch "cifar_${short}_er500"   "$seed" --dataset seq-cifar10 --model er    --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs --buffer_size 500
      launch "cifar_${short}_er1000"  "$seed" --dataset seq-cifar10 --model er    --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs --buffer_size 1000
      launch "cifar_${short}_derpp500" "$seed" --dataset seq-cifar10 --model derpp --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs --buffer_size 500 --alpha 0.1 --beta 0.5
    done
  done
}

case "$DATASET" in
  mnist) run_mnist ;;
  cifar|cifar10) run_cifar ;;
  all)   run_mnist; run_cifar ;;
  *) echo "Unknown --dataset $DATASET" >&2; exit 1 ;;
esac

echo "Wiring ablation jobs dispatched (max-parallel=$MAX_PARALLEL)."
