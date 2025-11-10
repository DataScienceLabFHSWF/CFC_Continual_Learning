# Quick Reference - Parallel Benchmarking

## 🚀 Quick Commands

```bash
# 1. Quick test (5 min) - SGD + ER on seq-mnist
./run_benchmarks.sh quick

# 2. Replay methods (30 min) - 13 replay-based methods
./run_benchmarks.sh replay

# 3. Regularization (10 min) - EWC, SI, LwF
./run_benchmarks.sh regularization

# 4. Full benchmark (2-3 hours) - All methods x all datasets
./run_benchmarks.sh all

# 5. Visualize results
python visualize_results.py results/benchmark_*.json
```

## 📊 Available Methods (21 total)

**Baselines**: sgd, joint  
**Replay**: er, der, derpp, gdumb, gss, hal, icarl, mer, er_ace, xder, xder_ce, xder_rpc, fdr  
**Regularization**: ewc_on, si, lwf, lwf_mc  
**Architecture**: pnn, rpc  
**Distillation**: bic, lucir

## ⚙️ Hardware Config

- **GPUs**: 2x NVIDIA H200 NVL (144GB each)
- **Parallelism**: 6 experiments (3 per GPU)
- **Timeout**: 2 hours per experiment

## 📈 Output

Results saved to: `results/benchmark_YYYYMMDD_HHMMSS.json`  
Plots saved to: `results/plots/*.png` and `results/plots/summary_table.csv`

See **BENCHMARKING.md** for detailed documentation.
