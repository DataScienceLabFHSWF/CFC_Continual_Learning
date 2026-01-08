# CfC Continual Learning - Next Steps Plan

**Last Updated:** 2026-01-08
**Context:** TEP bugs are fixed. HOPE stability is fixed (no NaNs), but Catastrophic Forgetting is severe (0% past task accuracy). Standard benchmarks (MNIST) are currently running.

## 1. Immediate Action Plan (Upon Return)

### Step 1: Check Benchmark Status
Upon returning, check the status of the long-running benchmarks (MNIST MLP).
```bash
# Check active sessions
tmux ls

# Check progress report (run the monitor script to update BENCHMARK_STATUS.md)
./monitor_benchmarks.sh
```
*   **Goal:** Ensure ~90 MNIST runs are completing successfully.

### Step 2: HOPE Status (Paused)
**Investigation Complete (Jan 8, 2026):**
*   Debugging validated stability fixes but confirmed architectural limitations for Class-IL.
*   **Outcome:** 0% accuracy on past tasks regardless of sparsity or simple consolidation strategies.
*   **Report:** See [docs/HOPE_IMPLEMENTATION_REPORT.md](docs/HOPE_IMPLEMENTATION_REPORT.md).
*   **Decision:** **Do NOT** launch parallel HOPE benchmarks. The architecture needs redesign (Buffer or Expansion).

## 2. Decision on Parallel HOPE Benchmarks

**Question:** *Could we start the HOPE benchmarks in parallel?*
**Answer:** **No.** 
*   **Reason:** The current implementation yields 0% accuracy on past tasks. Running 50-100 experiments now would result in 100 failed/useless logs. 
*   **Alternative:** Run a **Hyperparameter Search** instead. Create a config with varying `surprise_threshold` and `lr` values to find a configuration that retains knowledge.

## 3. Future Roadmap

### Phase A: Fix HOPE
1.  Debug `TitanMemory`.
2.  Validate on `seq-cifar10` until Class-IL > 10%.
3.  Once fixed, merge into `paper_benchmarks.yaml`.

### Phase B: Full Launch
1.  Once MNIST finishes (approx 2-3 days), launch CIFAR-10 benchmarks.
2.  Launch TEP benchmarks (now that TEP is fixed).

### Phase C: Analysis
1.  Run `scripts/analysis/analyze_paper_results.py`.
2.  Generate LaTeX tables.
