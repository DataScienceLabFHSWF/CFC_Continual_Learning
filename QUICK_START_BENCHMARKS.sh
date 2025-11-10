#!/bin/bash
# Quick Reference - Paper Benchmarks
# Just the essential commands you'll need

echo "=================================="
echo "CfC Paper Benchmarks - Quick Ref"
echo "=================================="
echo ""

echo "START BENCHMARKS:"
echo "  All datasets:  ./scripts/benchmarks/run_paper_benchmarks.sh --dataset all --max-parallel 4"
echo "  MNIST only:    ./scripts/benchmarks/run_paper_benchmarks.sh --dataset mnist --max-parallel 4"
echo "  CIFAR-10:      ./scripts/benchmarks/run_paper_benchmarks.sh --dataset cifar10 --max-parallel 4"
echo "  TEP only:      ./scripts/benchmarks/run_paper_benchmarks.sh --dataset tep --max-parallel 4"
echo ""

echo "MONITOR:"
echo "  List sessions: tmux ls"
echo "  Attach:        tmux attach -t paper_mnist_cfc_er200_s0"
echo "  Detach:        Ctrl+b, then d"
echo "  View log:      tail -f paper_results/logs/mnist_cfc_er200_seed0.log"
echo "  Count done:    ls paper_results/logs/*.log | wc -l"
echo ""

echo "ANALYZE:"
echo "  Generate:      python scripts/analysis/analyze_paper_results.py"
echo "  Visualize:     python scripts/analysis/visualize_results.py"
echo "  View summary:  cat paper_results/analysis/summary_mnist.csv"
echo "  LaTeX tables:  cat paper_results/analysis/table_mnist.tex"
echo ""

echo "RESULTS LOCATION:"
echo "  Logs:          paper_results/logs/*.log"
echo "  CSV data:      paper_results/*.csv"
echo "  Analysis:      paper_results/analysis/"
echo ""

echo "RUNTIME ESTIMATES (4 GPUs):"
echo "  MNIST:         ~56 hours  (90 runs)"
echo "  CIFAR-10:      ~150 hours (60 runs)"
echo "  TEP:           ~60 hours  (48 runs)"
echo "  Total:         ~266 hours (11 days)"
echo ""

echo "RECOMMENDED ORDER:"
echo "  1. Run MNIST (fast, validates infrastructure)"
echo "  2. Analyze MNIST results"
echo "  3. Run TEP (novel application)"
echo "  4. Run CIFAR-10 (longest, standard benchmark)"
echo ""
