#!/bin/bash
# ============================================================================
# Detached HOPE Benchmark Launcher
# ============================================================================
# This script launches the HOPE benchmarks in a persistent tmux session.
#
# Usage:
#   ./launch_hope_benchmark.sh
#
# The main orchestrator runs in: tmux session "hope_benchmark_orchestrator"
# ============================================================================

set -e

WORKSPACE="/home/fneubuerger/CFC_Continual_Learning"
CONFIG="configs/hope_suite.yaml"
PARALLEL=4

# Kill existing orchestrator if it exists
tmux kill-session -t hope_benchmark_orchestrator 2>/dev/null || true

echo "============================================================================"
echo "Launching HOPE Benchmarks in Detached Mode"
echo "============================================================================"
echo ""
echo "Config:       $CONFIG"
echo "Parallelism:  $PARALLEL"
echo ""
echo "The benchmarks will run in a detached tmux session."
echo "You can safely disconnect - everything will keep running."
echo ""

# Create the orchestrator session
tmux new-session -d -s hope_benchmark_orchestrator "
  cd $WORKSPACE
  source .venv/bin/activate
  
  echo '============================================================================'
  echo 'HOPE Benchmark Orchestrator'
  echo '============================================================================'
  echo ''
  echo 'Started: $(date)'
  echo 'Config: $CONFIG'
  echo ''
  echo 'Press Ctrl+b, then d to detach safely.'
  echo '============================================================================'
  echo ''
  
  # Run the benchmark runner
  python scripts/benchmarks/benchmark_runner.py --config $CONFIG --parallel $PARALLEL
  
  echo ''
  echo '============================================================================'
  echo 'All benchmarks finished!'
  echo 'Completed: $(date)'
  echo '============================================================================'
  echo ''
  echo 'This orchestrator session will remain open.'
  echo 'Press enter to close this session...'
  read
"

echo "✅ Orchestrator launched in tmux session: hope_benchmark_orchestrator"
echo ""
echo "============================================================================"
echo "How to Monitor:"
echo "============================================================================"
echo ""
echo "1. Attach to orchestrator:"
echo "   tmux attach -t hope_benchmark_orchestrator"
echo ""
echo "2. List all running experiments:"
echo "   tmux ls"
echo ""
echo "3. View logs:"
echo "   tail -f benchmark_results/run_*/summary.json"
echo ""
echo "============================================================================"
echo "Detach from any tmux session: Ctrl+b, then d"
echo "============================================================================"
echo ""

# Give tmux a moment to start
sleep 2

# Show the orchestrator output
echo "Current orchestrator status:"
tmux capture-pane -t hope_benchmark_orchestrator -p | tail -20
