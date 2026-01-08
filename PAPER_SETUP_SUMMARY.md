# CfC Continual Learning - Paper Benchmark Setup Complete

## Summary

This document summarizes the comprehensive paper benchmark infrastructure created for the CfC continual learning project.

## Status Update (2026-01-08)

**Benchmarks are Active.**
- **TEP**: Fixed critical data loading bugs. Ready for full deployment.
- **HOPE (Nested Learning)**: Stability fixes applied (NaNs resolved). Currently debugging catastrophic forgetting (0% Class-IL accuracy).
- **Status Dashboard**: See [BENCHMARK_STATUS.md](BENCHMARK_STATUS.md) for live progress.

## What Was Created

### 1. Paper Benchmark Configuration (`configs/paper_benchmarks.yaml`)

Comprehensive YAML configuration defining:
- **MNIST**: 30 configurations (10 MLP baseline + 10 CfC) × 3 seeds = 90 runs
- **CIFAR-10**: 20 configurations (10 ResNet + 10 CfC-CNN) × 3 seeds = 60 runs
- **TEP**: 16 configurations (8 CfC + 8 LSTM) × 3 seeds = 48 runs
- **Total**: 198 experiment runs

Models tested: 
- SGD, Joint (upper bound)
- ER (buffer: 200, 500, 1000)
- DER++ (buffer: 200, 500, 1000)
- ER-ACE (buffer: 200)
- A-GEM, GEM (buffer: 200)
- EWC-Online, SI, LwF

Epochs:
- MNIST: 10 epochs (~2-3 hours each)
- CIFAR-10: 50 epochs (~8-12 hours each)
- TEP: 20 epochs (~4-6 hours each)

### 2. Parallel Execution Script (`scripts/benchmarks/run_paper_benchmarks.sh`)

Sophisticated tmux-based parallel execution system:
- Launches experiments in separate tmux sessions
- Controls parallelism (default: 4 concurrent experiments)
- Logs to `paper_results/logs/`
- Saves CSV results to `paper_results/`
- Supports dataset filtering (`--dataset mnist|cifar10|tep|all`)
- Dry-run mode for testing (`--dry-run`)

Usage:
```bash
# Run all benchmarks
./scripts/benchmarks/run_paper_benchmarks.sh --dataset all --max-parallel 4

# Run specific dataset
./scripts/benchmarks/run_paper_benchmarks.sh --dataset mnist --max-parallel 4

# Preview commands
./scripts/benchmarks/run_paper_benchmarks.sh --dataset mnist --dry-run
```

### 3. Results Analysis Script (`scripts/analysis/analyze_paper_results.py`)

Python script for comprehensive result analysis:
- Parses log files and CSV results
- Aggregates across seeds (mean ± std)
- Generates summary tables per dataset
- Creates LaTeX tables ready for paper
- Identifies best-performing configurations
- Console output with statistics

Outputs:
- `paper_results/analysis/raw_results.csv` - All experiments
- `paper_results/analysis/summary_*.csv` - Per-dataset summaries
- `paper_results/analysis/table_*.tex` - LaTeX tables

Usage:
```bash
python scripts/analysis/analyze_paper_results.py \
    --results-dir paper_results \
    --output-dir paper_results/analysis
```

### 4. Repository Cleanup (`cleanup_repo.sh` - already executed)

Organized project structure:
```
CFC_Continual_Learning/
├── scripts/
│   ├── validation/          # Quick tests (1 epoch)
│   ├── benchmarks/          # Paper benchmarks
│   └── analysis/            # Result analysis
├── docs/                    # All documentation
├── results/                 # Organized results
├── configs/                 # Experiment configs
├── tests/                   # Test scripts
├── mammoth/                 # Mammoth v2.0
└── ncps/                    # Neural Circuit Policies
```

Removed:
- Duplicate validation scripts
- Old benchmark runners
- Scattered analysis scripts

### 5. Documentation (`PAPER_BENCHMARKS.md`)

Comprehensive 400+ line guide covering:
- Quick start instructions
- Detailed experiment configuration
- Monitoring and troubleshooting
- Expected results and metrics
- Analysis workflow
- Paper preparation workflow

## Expected Runtime

### Sequential (1 GPU)
- MNIST: 90 runs × 2.5 hours = 225 hours
- CIFAR-10: 60 runs × 10 hours = 600 hours
- TEP: 48 runs × 5 hours = 240 hours
- **Total: ~1065 hours (44 days)**

### Parallel (4 GPUs)
- MNIST: 225 / 4 = ~56 hours
- CIFAR-10: 600 / 4 = ~150 hours
- TEP: 240 / 4 = ~60 hours
- **Total: ~266 hours (11 days)**

### Parallel (8 GPUs)
- **Total: ~133 hours (5.5 days)**

## Recommended Workflow

### Phase 1: Quick Validation (Already Done)
```bash
./scripts/validation/run_full_validation_extended.sh
```
✅ Validates all backbones work (1 epoch each)

### Phase 2: MNIST Benchmarks (Start Here)
```bash
./scripts/benchmarks/run_paper_benchmarks.sh --dataset mnist --max-parallel 4
```
- Runtime: ~56 hours with 4 GPUs
- 90 experiments
- Fast iteration to verify infrastructure

### Phase 3: TEP Benchmarks
```bash
./scripts/benchmarks/run_paper_benchmarks.sh --dataset tep --max-parallel 4
```
- Runtime: ~60 hours with 4 GPUs
- 48 experiments
- Novel application (industrial fault detection)

### Phase 4: CIFAR-10 Benchmarks
```bash
./scripts/benchmarks/run_paper_benchmarks.sh --dataset cifar10 --max-parallel 4
```
- Runtime: ~150 hours with 4 GPUs
- 60 experiments
- Longest but standard vision benchmark

### Phase 5: Analysis
```bash
python scripts/analysis/analyze_paper_results.py
python scripts/analysis/visualize_results.py
```
- Generate all tables and plots
- Extract LaTeX tables
- Identify key findings

## Key Metrics

Each experiment reports:
1. **Class-IL Accuracy**: Average accuracy on all seen tasks (harder)
2. **Task-IL Accuracy**: Accuracy with task identity known (easier)
3. **Forgetting**: Performance drop on old tasks
4. **Convergence Ratio**: incremental_acc / joint_acc

## Monitoring

### Active Sessions
```bash
tmux ls
```

### Attach to Experiment
```bash
tmux attach -t paper_mnist_cfc_er200_s0
```

### View Logs
```bash
tail -f paper_results/logs/mnist_cfc_er200_seed0.log
```

### Check Progress
```bash
ls paper_results/logs/*.log | wc -l  # Count completed
```

## Configuration Files

All hyperparameters are version-controlled:
- `configs/paper_benchmarks.yaml` - High-level config
- `scripts/benchmarks/run_paper_benchmarks.sh` - Execution details
- `mammoth/utils/best_args.py` - Mammoth defaults

## Expected Results

### MNIST
- **Joint (upper bound)**: ~99.8%
- **Good CL methods**: >95% Class-IL
- **CfC hypothesis**: Better temporal modeling → less forgetting

### CIFAR-10
- **Joint (upper bound)**: ~90-95%
- **Good CL methods**: >70% Class-IL
- **CfC hypothesis**: Competitive with fewer parameters

### TEP
- **Challenge**: 22 tasks, temporal sequences
- **CfC hypothesis**: Temporal dynamics → better fault detection
- **Comparison**: CfC vs LSTM baseline

## Next Steps

1. **Start MNIST benchmarks** (recommended first):
   ```bash
   ./scripts/benchmarks/run_paper_benchmarks.sh --dataset mnist --max-parallel 4
   ```

2. **Monitor progress** (~2-3 days):
   ```bash
   watch -n 60 "ls paper_results/logs/*.log | wc -l"
   ```

3. **Analyze MNIST results**:
   ```bash
   python scripts/analysis/analyze_paper_results.py
   ```

4. **Proceed to TEP/CIFAR** based on MNIST insights

## Files Created

1. `configs/paper_benchmarks.yaml` - Experiment configuration
2. `scripts/benchmarks/run_paper_benchmarks.sh` - Parallel runner
3. `scripts/analysis/analyze_paper_results.py` - Result analyzer
4. `cleanup_repo.sh` - Repository organizer (executed)
5. `PAPER_BENCHMARKS.md` - User guide
6. `PAPER_SETUP_SUMMARY.md` - This file
7. Updated `README.md` - Main documentation

## Repository Status

✅ Clean, organized structure
✅ All scripts in `scripts/` subdirectories
✅ All docs in `docs/` directory
✅ Results organized in `results/`
✅ Comprehensive documentation
✅ Ready for paper-quality benchmarks

## Timeline Estimate

Assuming 4 GPUs running 24/7:

- **Week 1-2**: MNIST benchmarks (56 hours)
- **Week 2-4**: TEP benchmarks (60 hours)
- **Week 4-10**: CIFAR-10 benchmarks (150 hours)
- **Week 11**: Analysis and visualization
- **Total**: ~2.5 months for complete benchmark suite

Can be accelerated with more GPUs or running datasets in parallel.

## Questions?

See:
- `PAPER_BENCHMARKS.md` for detailed guide
- `README.md` for project overview
- `docs/MAMMOTH_VERSION.md` for Mammoth v2.0 details

---

**Created**: 2025-11-10
**Status**: Ready to run paper benchmarks
**Next Action**: Execute MNIST benchmarks
