#!/bin/bash
# ============================================================================
# Monitor HOPE Benchmarks
# ============================================================================
# Quick dashboard to monitor running HOPE benchmarks
# ============================================================================

BENCHMARK_DIR="/home/fneubuerger/CFC_Continual_Learning/benchmark_results"

# Find the latest run directory
LATEST_RUN=$(ls -td "$BENCHMARK_DIR"/run_* 2>/dev/null | head -1)

clear
echo "============================================================================"
echo "HOPE Benchmark Monitor - $(date)"
echo "============================================================================"
echo ""

if [ -z "$LATEST_RUN" ]; then
    echo "No benchmark runs found in $BENCHMARK_DIR"
    exit 1
fi

RUN_NAME=$(basename "$LATEST_RUN")
echo "Monitoring Run: $RUN_NAME"
echo "Directory: $LATEST_RUN"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TMUX SESSION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if tmux has-session -t hope_benchmark_orchestrator 2>/dev/null; then
    echo "✅ Orchestrator session is RUNNING"
else
    echo "❌ Orchestrator session is NOT FOUND"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "LOG FILES (Live Streaming)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
LOG_COUNT=$(ls "$LATEST_RUN"/*.log 2>/dev/null | wc -l)
echo "Found $LOG_COUNT log files."
echo ""

if [ "$LOG_COUNT" -gt 0 ]; then
    echo "Latest 5 logs:"
    ls -lt "$LATEST_RUN"/*.log | head -5 | awk '{print $9, $5, $6, $7, $8}'
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "RECENT ACTIVITY (Tail of latest log)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    LATEST_LOG=$(ls -t "$LATEST_RUN"/*.log | head -1)
    echo "File: $(basename "$LATEST_LOG")"
    echo ""
    tail -n 15 "$LATEST_LOG"
else
    echo "No log files created yet. Benchmarks might be initializing..."
fi

echo ""
echo "============================================================================"
echo "To watch continuously:  watch -n 5 ./monitor_hope.sh"
echo "============================================================================"
