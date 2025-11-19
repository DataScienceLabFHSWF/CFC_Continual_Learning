#!/bin/bash
# ============================================================================
# Launch Full Paper Benchmarks with Metrics
# ============================================================================
# This script launches comprehensive benchmarks for the paper, including:
# - All baseline methods (MLP, ResNet, RNN)
# - All CfC/NCP variants
# - All ablations (LTC, Random Sparse)
# - Advanced metrics enabled (RepStab, Grad Interference, Tau monitoring)
#
# Runs 16 experiments in parallel across available GPUs.
# ============================================================================

set -e

cd "$(dirname "$0")"
WORKSPACE="/home/fneubuerger/CFC_Continual_Learning"

echo "════════════════════════════════════════════════════════════════"
echo "  Full Paper Benchmarks with Metrics"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Config: configs/full_paper_benchmarks.yaml"
echo "  - Parallel runs: 16"
echo "  - Advanced metrics: ENABLED"
echo "  - Tau monitoring: ENABLED (auto for LTC backbones)"
echo "  - WandB logging: ENABLED"
echo ""
echo "Expected experiments:"
echo "  MNIST:    6 backbones × 6 methods × 3 seeds = 108 runs"
echo "  CIFAR-10: 4 backbones × 5 methods × 3 seeds = 60 runs"
echo "  TEP:      4 backbones × 4 methods × 3 seeds = 48 runs"
echo "  ────────────────────────────────────────────────────"
echo "  TOTAL:    216 experiments"
echo ""
echo "Estimated time:"
echo "  - MNIST: ~2 hours (10 epochs each)"
echo "  - CIFAR-10: ~24 hours (50 epochs each)"
echo "  - TEP: ~6 hours (20 epochs each)"
echo "  ────────────────────────────────────────────────────"
echo "  TOTAL: ~32 hours with 16 parallel runs"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

read -p "Do you want to proceed? (yes/no): " CONFIRM
if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Cleaning previous results..."
rm -rf "$WORKSPACE/paper_results/full_benchmarks"
mkdir -p "$WORKSPACE/paper_results/full_benchmarks"
mkdir -p "$WORKSPACE/paper_checkpoints/full_benchmarks"

echo ""
echo "Launching orchestrator in tmux session 'paper_full_benchmarks'..."
echo ""

tmux new-session -d -s paper_full_benchmarks "
cd $WORKSPACE && \\
source .venv/bin/activate && \\
./scripts/benchmarks/run_paper_benchmarks.sh \\
    --dataset all \\
    --max-parallel 16 \\
    --force
"

echo "════════════════════════════════════════════════════════════════"
echo "  Benchmarks launched!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Monitor progress:"
echo "  tmux attach -t paper_full_benchmarks"
echo ""
echo "Check running experiments:"
echo "  tmux ls | grep paper_"
echo ""
echo "View logs:"
echo "  tail -f $WORKSPACE/paper_results/full_benchmarks/logs/*.log"
echo ""
echo "Monitor with WandB:"
echo "  https://wandb.ai/fneubuerger/mammoth"
echo ""
echo "════════════════════════════════════════════════════════════════"
