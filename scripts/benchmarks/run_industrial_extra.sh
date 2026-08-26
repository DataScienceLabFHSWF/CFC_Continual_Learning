#!/usr/bin/env bash
# =============================================================================
# Additional industrial benchmarks: Steel Plates Faults, SECOM, C-MAPSS.
#
# Datasets:
#   steel-plates-faults  (3 tasks x 2 classes, static features)
#   secom                (3 tasks x 2 classes, static features, time-block split)
#   cmapss               (4 tasks x 3 classes, 30-step sliding windows)
#
# Backbones: <dataset>mlp (baseline) and <dataset>cfc (AutoNCP CfC).
# Protocols: SGD, Joint, ER@200, ER@500, DER++@500.
# Seeds:     5 (0..4).
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
PREFIX="ind_"
DATASET_FILTER="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=true; shift ;;
    --seeds)        IFS=',' read -ra SEEDS <<<"$2"; shift 2 ;;
    --dataset)      DATASET_FILTER="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
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
      --wandb_entity '$WANDB_ENTITY' --wandb_project '$WANDB_PROJECT' \
      --wandb_name '${exp}_seed${seed}' \
      2>&1 | tee '$lf'
  "
  echo "  start $sess (log: $lf)"
}

run_dataset() {
  local dataset="$1" short="$2" n_epochs="$3" lr="$4" bs="$5"
  for bb in "${short}mlp" "${short}cfc"; do
    local bbtag=${bb//[^a-zA-Z0-9]/}
    for seed in "${SEEDS[@]}"; do
      launch "${short}_${bbtag}_sgd"      "$seed" --dataset "$dataset" --model sgd   --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs
      launch "${short}_${bbtag}_joint"    "$seed" --dataset "$dataset" --model joint --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs
      launch "${short}_${bbtag}_er200"    "$seed" --dataset "$dataset" --model er    --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs --buffer_size 200
      launch "${short}_${bbtag}_er500"    "$seed" --dataset "$dataset" --model er    --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs --buffer_size 500
      launch "${short}_${bbtag}_derpp500" "$seed" --dataset "$dataset" --model derpp --backbone "$bb" --n_epochs $n_epochs --lr $lr --batch_size $bs --buffer_size 500 --alpha 0.1 --beta 0.5
    done
  done
}

case "$DATASET_FILTER" in
  steel-plates-faults|steel) run_dataset steel-plates-faults steelplates 30 0.01 32 ;;
  secom)                     run_dataset secom secom 30 0.01 32 ;;
  cmapss)                    run_dataset cmapss cmapss 10 0.005 32 ;;
  all)
    run_dataset steel-plates-faults steelplates 30 0.01 32
    run_dataset secom secom 30 0.01 32
    run_dataset cmapss cmapss 10 0.005 32
    ;;
  *) echo "Unknown --dataset $DATASET_FILTER" >&2; exit 1 ;;
esac

echo "Industrial benchmark jobs dispatched (max-parallel=$MAX_PARALLEL)."
