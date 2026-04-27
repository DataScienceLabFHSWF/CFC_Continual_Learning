#!/usr/bin/env bash
# =============================================================================
# Tennessee Eastman Process (redesigned 11-task / 2-classes-per-task split).
#
# Backbones: tepcfc, tepltc, teplstm, tep_dense_cfc, tep_random_sparse
#            (tepmlp omitted: not registered; can be added later)
# Protocols: SGD, Joint, ER@200, ER@500, DER++@500
# Seeds:     5 (0..4)  -> matches the n=5 main-results bump
# =============================================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/home/fneubuerger/CFC_Continual_Learning}"
MAMMOTH_DIR="$WORKSPACE/mammoth"
LOG_DIR="$WORKSPACE/paper_results/logs"
WANDB_ENTITY="${WANDB_ENTITY:-fneubuerger}"
WANDB_PROJECT="${WANDB_PROJECT:-mammoth}"
SEEDS=(0 1 2 3 4)
MAX_PARALLEL=2
DRY_RUN=false
PREFIX="tep_"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=true; shift ;;
    --seeds)        IFS=',' read -ra SEEDS <<<"$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
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
    n=$(tmux list-sessions 2>/dev/null | grep -c "^${PREFIX}" || true)
    [[ -z "$n" ]] && n=0
    (( n < MAX_PARALLEL )) && return
    sleep 10
  done
}

launch() {
  local exp="$1"; local seed="$2"; shift 2
  local lf="$LOG_DIR/${exp}_seed${seed}.log"
  if is_done "$lf"; then echo "  skip $exp seed=$seed (done)"; return; fi
  if $DRY_RUN; then echo "[DRY] $exp seed=$seed :: $*"; return; fi
  wait_for_slot
  local sess="${PREFIX}${exp}_s${seed}"
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

# Backbones to evaluate on the 11x2 redesigned TEP split.
BACKBONES=(tepcfc tepltc teplstm tep_dense_cfc tep_random_sparse)

N_EPOCHS=20
LR=0.001
BS=32

for bb in "${BACKBONES[@]}"; do
  short=${bb//[^a-zA-Z0-9]/}
  for seed in "${SEEDS[@]}"; do
    launch "tep_${short}_sgd"      "$seed" --dataset tennessee-eastman --model sgd   --backbone "$bb" --n_epochs $N_EPOCHS --lr $LR --batch_size $BS
    launch "tep_${short}_joint"    "$seed" --dataset tennessee-eastman --model joint --backbone "$bb" --n_epochs $N_EPOCHS --lr $LR --batch_size $BS
    launch "tep_${short}_er200"    "$seed" --dataset tennessee-eastman --model er    --backbone "$bb" --n_epochs $N_EPOCHS --lr $LR --batch_size $BS --buffer_size 200
    launch "tep_${short}_er500"    "$seed" --dataset tennessee-eastman --model er    --backbone "$bb" --n_epochs $N_EPOCHS --lr $LR --batch_size $BS --buffer_size 500
    launch "tep_${short}_derpp500" "$seed" --dataset tennessee-eastman --model derpp --backbone "$bb" --n_epochs $N_EPOCHS --lr $LR --batch_size $BS --buffer_size 500 --alpha 0.1 --beta 0.5
  done
done

echo "TEP redesigned-split jobs dispatched (backbones=${#BACKBONES[@]}, seeds=${#SEEDS[@]}, max-parallel=$MAX_PARALLEL)."
