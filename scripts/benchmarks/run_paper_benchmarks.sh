#!/bin/bash
# ============================================================================
# CfC Continual Learning - Paper Benchmark Runner
# ============================================================================
# This script runs comprehensive benchmarks for the CfC continual learning paper.
# It executes multiple experiments in parallel using tmux sessions.
#
# Usage:
#   ./run_paper_benchmarks.sh [--dataset DATASET] [--dry-run] [--force]
#
# Options:
#   --dataset DATASET   Run only specific dataset (mnist, cifar10, tep, all)
#   --dry-run           Print commands without executing
#   --force             Force re-run even if experiments already completed
#   --max-parallel N    Maximum parallel experiments (default: 4)
# ============================================================================

set -e

# Configuration
WORKSPACE="/home/fneubuerger/CFC_Continual_Learning"
MAMMOTH_DIR="$WORKSPACE/mammoth"
RESULTS_DIR="$WORKSPACE/paper_results"
CHECKPOINT_DIR="$WORKSPACE/paper_checkpoints"
LOG_DIR="$RESULTS_DIR/logs"

# WandB configuration (loaded from .secrets.json)
WANDB_ENTITY="fneubuerger"
WANDB_PROJECT="mammoth"

# Default settings
DATASET="all"
DRY_RUN=false
FORCE_RERUN=false
MAX_PARALLEL=4
SEEDS=(0 1 2)

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --force)
      FORCE_RERUN=true
      shift
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# Check if experiment already completed
is_experiment_completed() {
  local exp_name=$1
  local seed=$2
  local log_file="$LOG_DIR/${exp_name}_seed${seed}.log"
  
  # If force rerun, always return false (not completed)
  if [ "$FORCE_RERUN" = true ]; then
    return 1
  fi
  
  # Check if log file exists and contains completion marker
  if [ -f "$log_file" ]; then
    if grep -q "Experiment completed:" "$log_file" 2>/dev/null; then
      return 0  # Completed
    fi
  fi
  return 1  # Not completed
}

# Helper function to run experiment
run_experiment() {
  local exp_name=$1
  local dataset=$2
  local model=$3
  local backbone=$4
  local n_epochs=$5
  local lr=$6
  local batch_size=$7
  local seed=$8
  local extra_args=$9
  
  local session_name="paper_${exp_name}_s${seed}"
  local log_file="$LOG_DIR/${exp_name}_seed${seed}.log"
  local result_file="$RESULTS_DIR/${exp_name}_seed${seed}.csv"
  
  # Check if already completed
  if is_experiment_completed "$exp_name" "$seed"; then
    echo "  Skipped: $exp_name (seed $seed) - already completed"
    return
  fi
  
  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would run: $exp_name (seed $seed)"
    return
  fi
  
  # Kill existing session if it exists
  tmux kill-session -t "$session_name" 2>/dev/null || true
  
  # Create new session
  tmux new-session -d -s "$session_name" "
    cd $MAMMOTH_DIR
    source $WORKSPACE/.venv/bin/activate
    
    # Load WandB credentials
    export WANDB_API_KEY=\$(cat $WORKSPACE/.secrets.json | grep wandb_api_key | cut -d'\"' -f4)
    
    echo '========================================'
    echo 'Experiment: $exp_name'
    echo 'Seed: $seed'
    echo 'Started: \$(date)'
    echo 'WandB: $WANDB_ENTITY/$WANDB_PROJECT'
    echo '========================================'
    
    python utils/main.py \\
      --dataset $dataset \\
      --model $model \\
      --backbone $backbone \\
      --n_epochs $n_epochs \\
      --lr $lr \\
      --batch_size $batch_size \\
      --seed $seed \\
      --num_workers 4 \\
      --wandb_entity $WANDB_ENTITY \\
      --wandb_project $WANDB_PROJECT \\
      --wandb_name ${exp_name}_seed${seed} \\
      $extra_args \\
      2>&1 | tee $log_file
    
    echo ''
    echo '========================================'
    echo 'Experiment completed: \$(date)'
    echo '========================================'
    
    read -p 'Press enter to close...'
  "
  
  echo "  Started: $session_name (log: $log_file)"
}

# Wait for available slot
wait_for_slot() {
  while true; do
    # Count only experiment sessions (paper_*), not the orchestrator (benchmark_orchestrator)
    local running=$(tmux list-sessions 2>/dev/null | grep "^paper_" | wc -l || echo 0)
    if [ -z "$running" ] || [ "$running" = "" ]; then
      running=0
    fi
    if [ "$running" -lt "$MAX_PARALLEL" ]; then
      return
    fi
    sleep 10
  done
}

# ============================================================================
# MNIST Experiments
# ============================================================================
run_mnist_experiments() {
  echo ""
  echo "========================================"
  echo "MNIST Experiments"
  echo "========================================"
  
  local N_EPOCHS=10
  local BATCH_SIZE=32
  
  # MLP Baseline - SGD
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_mlp_sgd" "seq-mnist" "sgd" "mnistmlp" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed ""
  done
  
  # MLP Baseline - Joint (upper bound)
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_mlp_joint" "seq-mnist" "joint" "mnistmlp" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed ""
  done
  
  # MLP Baseline - ER (buffer 200)
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_mlp_er200" "seq-mnist" "er" "mnistmlp" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size 200"
  done
  
  # MLP Baseline - ER (buffer 500)
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_mlp_er500" "seq-mnist" "er" "mnistmlp" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size 500"
  done
  
  # MLP Baseline - DER++ (buffer 200)
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_mlp_derpp200" "seq-mnist" "derpp" "mnistmlp" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size 200 --alpha 0.1 --beta 0.5"
  done
  
  # MLP Baseline - DER++ (buffer 500)
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_mlp_derpp500" "seq-mnist" "derpp" "mnistmlp" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size 500 --alpha 0.1 --beta 0.5"
  done
  
  # MLP Baseline - ER-ACE (buffer 200)
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_mlp_erace200" "seq-mnist" "er_ace" "mnistmlp" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size 200"
  done
  
  # MLP Baseline - A-GEM
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_mlp_agem" "seq-mnist" "agem" "mnistmlp" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size 200"
  done
  
  # MLP Baseline - GEM
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_mlp_gem" "seq-mnist" "gem" "mnistmlp" \
      $N_EPOCHS 0.1 $BATCH_SIZE $seed "--buffer_size 200 --gamma 0.5"
  done
  
  # MLP Baseline - EWC
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_mlp_ewc" "seq-mnist" "ewc_on" "mnistmlp" \
      $N_EPOCHS 0.1 $BATCH_SIZE $seed "--e_lambda 0.7 --gamma 1.0"
  done
  
  # MLP Baseline - SI
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_mlp_si" "seq-mnist" "si" "mnistmlp" \
      $N_EPOCHS 0.1 $BATCH_SIZE $seed "--c 0.5 --xi 1.0"
  done
  
  # MLP Baseline - LwF
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_mlp_lwf" "seq-mnist" "lwf" "mnistmlp" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--alpha 1.0 --softmax_temp 2.0"
  done
  
  echo ""
  echo "----------------------------------------"
  echo "CfC Experiments"
  echo "----------------------------------------"
  
  # CfC - SGD
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_cfc_sgd" "seq-mnist" "sgd" "mnistcfc" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed ""
  done
  
  # CfC - Joint
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_cfc_joint" "seq-mnist" "joint" "mnistcfc" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed ""
  done
  
  # CfC - ER (buffer 200)
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_cfc_er200" "seq-mnist" "er" "mnistcfc" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size 200"
  done
  
  # CfC - ER (buffer 500)
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_cfc_er500" "seq-mnist" "er" "mnistcfc" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size 500"
  done
  
  # CfC - DER++ (buffer 200)
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_cfc_derpp200" "seq-mnist" "derpp" "mnistcfc" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size 200 --alpha 0.1 --beta 0.5"
  done
  
  # CfC - DER++ (buffer 500)
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_cfc_derpp500" "seq-mnist" "derpp" "mnistcfc" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size 500 --alpha 0.1 --beta 0.5"
  done
  
  # CfC - ER-ACE (buffer 200)
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_cfc_erace200" "seq-mnist" "er_ace" "mnistcfc" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size 200"
  done
  
  # CfC - A-GEM
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_cfc_agem" "seq-mnist" "agem" "mnistcfc" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size 200"
  done
  
  # CfC - EWC
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_cfc_ewc" "seq-mnist" "ewc_on" "mnistcfc" \
      $N_EPOCHS 0.1 $BATCH_SIZE $seed "--e_lambda 0.7 --gamma 1.0"
  done
  
  # CfC - SI
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "mnist_cfc_si" "seq-mnist" "si" "mnistcfc" \
      $N_EPOCHS 0.1 $BATCH_SIZE $seed "--c 0.5 --xi 1.0"
  done
}

# ============================================================================
# CIFAR-10 Experiments
# ============================================================================
run_cifar_experiments() {
  echo ""
  echo "========================================"
  echo "CIFAR-10 Experiments"
  echo "========================================"
  
  local N_EPOCHS=50
  local BATCH_SIZE=32
  
  # ResNet18 Baseline - SGD
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "cifar_resnet_sgd" "seq-cifar10" "sgd" "resnet18" \
      $N_EPOCHS 0.1 $BATCH_SIZE $seed ""
  done
  
  # ResNet18 Baseline - Joint
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "cifar_resnet_joint" "seq-cifar10" "joint" "resnet18" \
      $N_EPOCHS 0.1 $BATCH_SIZE $seed ""
  done
  
  # ResNet18 - ER (multiple buffer sizes)
  for buffer in 200 500 1000; do
    for seed in "${SEEDS[@]}"; do
      wait_for_slot
      run_experiment "cifar_resnet_er${buffer}" "seq-cifar10" "er" "resnet18" \
        $N_EPOCHS 0.1 $BATCH_SIZE $seed "--buffer_size $buffer"
    done
  done
  
  # ResNet18 - DER++ (multiple buffer sizes)
  for buffer in 200 500 1000; do
    for seed in "${SEEDS[@]}"; do
      wait_for_slot
      run_experiment "cifar_resnet_derpp${buffer}" "seq-cifar10" "derpp" "resnet18" \
        $N_EPOCHS 0.1 $BATCH_SIZE $seed "--buffer_size $buffer --alpha 0.2 --beta 0.5"
    done
  done
  
  # ResNet18 - ER-ACE (multiple buffer sizes)
  for buffer in 200 500 1000; do
    for seed in "${SEEDS[@]}"; do
      wait_for_slot
      run_experiment "cifar_resnet_erace${buffer}" "seq-cifar10" "er_ace" "resnet18" \
        $N_EPOCHS 0.1 $BATCH_SIZE $seed "--buffer_size $buffer"
    done
  done
  
  # ResNet18 - EWC
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "cifar_resnet_ewc" "seq-cifar10" "ewc_on" "resnet18" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--e_lambda 1.0 --gamma 1.0"
  done
  
  # ResNet18 - SI
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "cifar_resnet_si" "seq-cifar10" "si" "resnet18" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed "--c 1.0 --xi 1.0"
  done
  
  echo ""
  echo "----------------------------------------"
  echo "CfC CNN Experiments"
  echo "----------------------------------------"
  
  # CfC CNN - SGD
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "cifar_cfc_sgd" "seq-cifar10" "sgd" "cnn-cfc" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed ""
  done
  
  # CfC CNN - Joint
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "cifar_cfc_joint" "seq-cifar10" "joint" "cnn-cfc" \
      $N_EPOCHS 0.03 $BATCH_SIZE $seed ""
  done
  
  # CfC CNN - ER (multiple buffer sizes)
  for buffer in 200 500 1000; do
    for seed in "${SEEDS[@]}"; do
      wait_for_slot
      run_experiment "cifar_cfc_er${buffer}" "seq-cifar10" "er" "cnn-cfc" \
        $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size $buffer"
    done
  done
  
  # CfC CNN - DER++ (multiple buffer sizes)
  for buffer in 200 500 1000; do
    for seed in "${SEEDS[@]}"; do
      wait_for_slot
      run_experiment "cifar_cfc_derpp${buffer}" "seq-cifar10" "derpp" "cnn-cfc" \
        $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size $buffer --alpha 0.1 --beta 0.5"
    done
  done
  
  # CfC CNN - ER-ACE (multiple buffer sizes)
  for buffer in 200 500 1000; do
    for seed in "${SEEDS[@]}"; do
      wait_for_slot
      run_experiment "cifar_cfc_erace${buffer}" "seq-cifar10" "er_ace" "cnn-cfc" \
        $N_EPOCHS 0.03 $BATCH_SIZE $seed "--buffer_size $buffer"
    done
  done
  
  # CfC CNN - EWC
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "cifar_cfc_ewc" "seq-cifar10" "ewc_on" "cnn-cfc" \
      $N_EPOCHS 0.01 $BATCH_SIZE $seed "--e_lambda 0.7 --gamma 1.0"
  done
}

# ============================================================================
# TEP Experiments
# ============================================================================
run_tep_experiments() {
  echo ""
  echo "========================================"
  echo "Tennessee Eastman Process Experiments"
  echo "========================================"
  
  local N_EPOCHS=20
  local BATCH_SIZE=32
  
  # CfC - SGD
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "tep_cfc_sgd" "tennessee-eastman" "sgd" "tepcfc" \
      $N_EPOCHS 0.001 $BATCH_SIZE $seed ""
  done
  
  # CfC - ER (multiple buffer sizes)
  for buffer in 200 500 1000; do
    for seed in "${SEEDS[@]}"; do
      wait_for_slot
      run_experiment "tep_cfc_er${buffer}" "tennessee-eastman" "er" "tepcfc" \
        $N_EPOCHS 0.001 $BATCH_SIZE $seed "--buffer_size $buffer"
    done
  done
  
  # CfC - DER++ (multiple buffer sizes)
  for buffer in 200 500 1000; do
    for seed in "${SEEDS[@]}"; do
      wait_for_slot
      run_experiment "tep_cfc_derpp${buffer}" "tennessee-eastman" "derpp" "tepcfc" \
        $N_EPOCHS 0.001 $BATCH_SIZE $seed "--buffer_size $buffer --alpha 0.1 --beta 0.5"
    done
  done
  
  # CfC - EWC
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "tep_cfc_ewc" "tennessee-eastman" "ewc_on" "tepcfc" \
      $N_EPOCHS 0.001 $BATCH_SIZE $seed "--e_lambda 0.5 --gamma 1.0"
  done
  
  echo ""
  echo "----------------------------------------"
  echo "LSTM Baseline Experiments"
  echo "----------------------------------------"
  
  # LSTM - SGD
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "tep_lstm_sgd" "tennessee-eastman" "sgd" "teplstm" \
      $N_EPOCHS 0.001 $BATCH_SIZE $seed ""
  done
  
  # LSTM - ER (multiple buffer sizes)
  for buffer in 200 500 1000; do
    for seed in "${SEEDS[@]}"; do
      wait_for_slot
      run_experiment "tep_lstm_er${buffer}" "tennessee-eastman" "er" "teplstm" \
        $N_EPOCHS 0.001 $BATCH_SIZE $seed "--buffer_size $buffer"
    done
  done
  
  # LSTM - DER++ (multiple buffer sizes)
  for buffer in 200 500 1000; do
    for seed in "${SEEDS[@]}"; do
      wait_for_slot
      run_experiment "tep_lstm_derpp${buffer}" "tennessee-eastman" "derpp" "teplstm" \
        $N_EPOCHS 0.001 $BATCH_SIZE $seed "--buffer_size $buffer --alpha 0.1 --beta 0.5"
    done
  done
  
  # LSTM - EWC
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    run_experiment "tep_lstm_ewc" "tennessee-eastman" "ewc_on" "teplstm" \
      $N_EPOCHS 0.001 $BATCH_SIZE $seed "--e_lambda 0.5 --gamma 1.0"
  done
}

# ============================================================================
# Main Execution
# ============================================================================
echo "============================================================================"
echo "CfC Continual Learning - Paper Benchmark Suite"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Workspace:      $WORKSPACE"
echo "  Results Dir:    $RESULTS_DIR"
echo "  Max Parallel:   $MAX_PARALLEL"
echo "  Seeds:          ${SEEDS[*]}"
echo "  Dataset Filter: $DATASET"
echo "  Dry Run:        $DRY_RUN"
echo ""

# Run experiments based on dataset filter
case $DATASET in
  mnist)
    run_mnist_experiments
    ;;
  cifar10|cifar)
    run_cifar_experiments
    ;;
  tep)
    run_tep_experiments
    ;;
  all)
    run_mnist_experiments
    run_cifar_experiments
    run_tep_experiments
    ;;
  *)
    echo "Error: Unknown dataset '$DATASET'"
    echo "Valid options: mnist, cifar10, tep, all"
    exit 1
    ;;
esac

# Wait for all experiments to complete
if [ "$DRY_RUN" = false ]; then
  echo ""
  echo "============================================================================"
  echo "All experiments launched!"
  echo "============================================================================"
  echo ""
  echo "Monitor progress:"
  echo "  tmux ls                           # List all sessions"
  echo "  tmux attach -t <session>          # Attach to session"
  echo "  tail -f $LOG_DIR/*.log            # View logs"
  echo ""
  echo "Results will be saved to:"
  echo "  $RESULTS_DIR/"
  echo ""
  echo "To generate summary:"
  echo "  ./analyze_paper_results.py"
  echo ""
fi
