#!/bin/bash
# Quick start script for running CL benchmarks

set -e

# Activate environment
source .venv/bin/activate

# Load secrets
if [ -f .secrets.json ]; then
    export WANDB_API_KEY=$(python -c "import json; print(json.load(open('.secrets.json'))['wandb_api_key'])")
fi

echo "=================================================="
echo "Continual Learning Benchmarking Suite"
echo "=================================================="
echo ""
echo "Available benchmarks:"
echo "  1. Replay methods (ER, DER, DER++, GDumb, etc.)"
echo "  2. Regularization methods (EWC, SI, LwF)"
echo "  3. All methods across all datasets (comprehensive)"
echo "  4. Quick test (SGD + ER on seq-mnist)"
echo ""

# Parse command line argument
BENCHMARK=${1:-all}

case $BENCHMARK in
  replay)
    echo "Running replay methods benchmark..."
    python benchmark_replay_methods.py
    ;;
  
  regularization)
    echo "Running regularization methods benchmark..."
    python benchmark_regularization_methods.py
    ;;
  
  all)
    echo "Running comprehensive benchmark (all methods x all datasets)..."
    python benchmark_all_methods.py
    ;;
  
  quick)
    echo "Running quick test benchmark..."
    python benchmark_all_methods.py --methods sgd er --datasets seq-mnist --seeds 1
    ;;
  
  custom)
    echo "Custom benchmark - pass your own arguments..."
    shift
    python benchmark_all_methods.py "$@"
    ;;
  
  *)
    echo "Usage: $0 {replay|regularization|all|quick|custom}"
    echo ""
    echo "Examples:"
    echo "  $0 replay                # Run only replay-based methods"
    echo "  $0 regularization        # Run only regularization methods"
    echo "  $0 all                   # Run comprehensive benchmark"
    echo "  $0 quick                 # Quick test with SGD and ER"
    echo "  $0 custom --methods sgd er ewc_on --datasets seq-mnist perm-mnist --seeds 5"
    exit 1
    ;;
esac

echo ""
echo "=================================================="
echo "Benchmark complete!"
echo "Results saved to: results/"
echo "=================================================="
