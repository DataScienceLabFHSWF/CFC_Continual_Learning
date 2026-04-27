#!/bin/bash
# ============================================================================
# Monitor Paper Benchmarks
# ============================================================================
# Quick dashboard to monitor all running benchmarks
# ============================================================================

RESULTS_DIR="/home/fneubuerger/CFC_Continual_Learning/paper_results"

clear
echo "============================================================================"
echo "Paper Benchmark Monitor - $(date)"
echo "============================================================================"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TMUX SESSIONS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tmux ls 2>/dev/null | grep "^paper_" | wc -l | xargs echo "Running experiments:"
echo ""
tmux ls 2>/dev/null | grep "^paper_" | head -10
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "LOG FILES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "$RESULTS_DIR/logs" ]; then
    ls -lh "$RESULTS_DIR/logs"/*.log 2>/dev/null | wc -l | xargs echo "Log files created:"
    echo ""
    echo "Latest 5 logs:"
    ls -lt "$RESULTS_DIR/logs"/*.log 2>/dev/null | head -5 | awk '{print $9, $5, $6, $7, $8}'
else
    echo "No logs directory yet"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "RECENT ACTIVITY (last 10 log lines)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "$RESULTS_DIR/logs" ]; then
    LATEST_LOG=$(ls -t "$RESULTS_DIR/logs"/*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "Latest log: $(basename $LATEST_LOG)"
        echo ""
        tail -10 "$LATEST_LOG" 2>/dev/null | grep -E "Accuracy|Task|Epoch|ERROR|WARNING" || echo "No recent activity"
    fi
else
    echo "No logs yet"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "QUICK COMMANDS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Attach to orchestrator:  tmux attach -t paper_orchestrator"
echo "  List all sessions:       tmux ls"
echo "  View specific log:       tail -f paper_results/logs/mnist_mlp_sgd_seed0.log"
echo "  WandB dashboard:         https://wandb.ai/fneubuerger/mammoth"
echo "  Refresh this monitor:    watch -n 30 ./monitor_benchmarks.sh"
echo ""
echo "Press Ctrl+C to exit"
echo "============================================================================"
