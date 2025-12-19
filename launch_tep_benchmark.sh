#!/bin/bash
# ============================================================================
# Detached TEP Benchmark Launcher
# ============================================================================
# This script launches the Tennessee Eastman Process benchmarks.
# ============================================================================

set -e

WORKSPACE="/home/fneubuerger/CFC_Continual_Learning"
CONFIG="configs/tep_benchmark.yaml"
PARALLEL=4

# Kill existing orchestrator if it exists
tmux kill-session -t tep_benchmark_orchestrator 2>/dev/null || true

echo "============================================================================"
echo "Launching TEP Benchmarks in Detached Mode"
echo "============================================================================"
echo ""
echo "Config:       $CONFIG"
echo "Parallelism:  $PARALLEL"
echo ""
echo "The benchmarks will run in a detached tmux session."
echo "You can safely disconnect - everything will keep running."
echo ""

# Create the orchestrator session
tmux new-session -d -s tep_benchmark_orchestrator "
  cd $WORKSPACE
  source .venv/bin/activate
  
  echo '============================================================================'
  echo 'TEP Benchmark Orchestrator'
  echo '============================================================================'
  echo ''
  echo 'Started: \$(date)'
  echo 'Config: $CONFIG'
  echo ''
  echo 'Press Ctrl+b, then d to detach safely.'
  echo '============================================================================'
  echo ''
  
  # Run the benchmark runner
  python scripts/benchmarks/benchmark_runner.py --config $CONFIG --parallel $PARALLEL
"

echo "✅ Orchestrator launched in tmux session: tep_benchmark_orchestrator"
echo ""
echo "Monitor with:"
echo "  tmux attach -t tep_benchmark_orchestrator"
echo "  ./monitor_tep.sh"
