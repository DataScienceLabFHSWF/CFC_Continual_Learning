#!/bin/bash
# ============================================================================
# Detached Paper Benchmark Launcher
# ============================================================================
# This script launches the paper benchmarks in a persistent tmux session
# that will survive VPN disconnects, SSH drops, and terminal closures.
#
# Usage:
#   ./launch_paper_benchmarks.sh [--dataset DATASET] [--max-parallel N]
#
# The main orchestrator runs in: tmux session "benchmark_orchestrator"
# Individual experiments run in:  tmux sessions "paper_*"
# ============================================================================

set -e

WORKSPACE="/home/fneubuerger/CFC_Continual_Learning"
DATASET="${1:-mnist}"
MAX_PARALLEL="${2:-4}"

# Kill existing orchestrator if it exists
tmux kill-session -t benchmark_orchestrator 2>/dev/null || true

echo "============================================================================"
echo "Launching Paper Benchmarks in Detached Mode"
echo "============================================================================"
echo ""
echo "Dataset:      $DATASET"
echo "Max Parallel: $MAX_PARALLEL"
echo ""
echo "The benchmarks will run in a detached tmux session."
echo "You can safely disconnect - everything will keep running."
echo ""

# Create the orchestrator session
tmux new-session -d -s benchmark_orchestrator "
  cd $WORKSPACE
  source .venv/bin/activate
  
  echo '============================================================================'
  echo 'Paper Benchmark Orchestrator'
  echo '============================================================================'
  echo ''
  echo 'Started: \$(date)'
  echo 'Dataset: $DATASET'
  echo 'Max Parallel: $MAX_PARALLEL'
  echo ''
  echo 'This session is managing all benchmark experiments.'
  echo 'Individual experiments run in separate tmux sessions (paper_*).'
  echo ''
  echo 'Press Ctrl+b, then d to detach safely.'
  echo '============================================================================'
  echo ''
  
  # Run the benchmark script
  ./scripts/benchmarks/run_paper_benchmarks.sh --dataset $DATASET --max-parallel $MAX_PARALLEL
  
  echo ''
  echo '============================================================================'
  echo 'All benchmarks launched!'
  echo 'Completed: \$(date)'
  echo '============================================================================'
  echo ''
  echo 'Monitor progress:'
  echo '  tmux ls                              # List all sessions'
  echo '  tmux attach -t paper_mnist_mlp_sgd_s0  # Attach to specific experiment'
  echo '  tail -f paper_results/logs/*.log     # View logs'
  echo ''
  echo 'This orchestrator session will remain open.'
  echo 'Press enter to close this session (experiments will continue)...'
  read
"

echo "✅ Orchestrator launched in tmux session: benchmark_orchestrator"
echo ""
echo "============================================================================"
echo "How to Monitor:"
echo "============================================================================"
echo ""
echo "1. Attach to orchestrator:"
echo "   tmux attach -t benchmark_orchestrator"
echo ""
echo "2. List all running experiments:"
echo "   tmux ls"
echo ""
echo "3. Attach to specific experiment:"
echo "   tmux attach -t paper_mnist_mlp_sgd_s0"
echo ""
echo "4. View logs:"
echo "   tail -f paper_results/logs/mnist_mlp_sgd_seed0.log"
echo ""
echo "5. Check WandB dashboard:"
echo "   https://wandb.ai/fneubuerger/mammoth"
echo ""
echo "============================================================================"
echo "Detach from any tmux session: Ctrl+b, then d"
echo "Everything will keep running even if you disconnect!"
echo "============================================================================"
echo ""

# Give tmux a moment to start
sleep 2

# Show the orchestrator output
echo "Current orchestrator status:"
tmux capture-pane -t benchmark_orchestrator -p | tail -20
