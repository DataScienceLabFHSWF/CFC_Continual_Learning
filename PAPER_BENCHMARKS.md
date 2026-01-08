# CfC Continual Learning - Paper Benchmarks Guide

**Status Note (2026-01-08):**
- **TEP**: Verified working.
- **HOPE**: **ARCHITECTURE FAILED CLASS-IL VALIDATION.**  
  *Analysis:* HOPE (Nested Learning/Titan Memory) is designed for **Language Modeling** (Fixed Vocabulary, Next Token Prediction). The memory modules learn dense representations $x \to z$ that map to a fixed output space $Y$. In **Class-Incremental Learning** (Mammoth benchmarks), the output head $Y$ grows (new classes added). The gradients from new classes cause the dense memory to shift rapidly ("Plasticity"), destroying keys for old classes ("Catastrophic Forgetting") because there is no Replay providing gradients for the old head connections.
  *Conclusion:* Pure HOPE is structurally incompatible with growing-head Class-IL without a Replay Buffer.

### 3. CfC vs. Catastrophic Forgetting (SGD Baseline)
**Hypothesis Verified:** Does the CfC architecture (Neural Circuit Policies) inherently mitigate catastrophic forgetting relative to ResNet when using only SGD (No Replay)?
**Result:** **NO.**
*   **Metric:** Class-IL Accuracy on Seq-CIFAR10
*   **ResNet-18 (SGD):** 19.63% (Chance/Last Task only)
*   **CfC (SGD):** 19.66% (Chance/Last Task only)
*   **Conclusion:** The sparsity and ODE dynamics of CfC do **not** provide immunity to catastrophic forgetting in the Class-IL setting. Like standard architectures, CfC requires a Replay Buffer (ER/DER) to maintain performance on past tasks.

- **See [BENCHMARK_STATUS.md](BENCHMARK_STATUS.md) for current run statistics.**

This guide explains how to run the comprehensive benchmark suite for the CfC continual learning paper.

## Overview

The benchmark suite evaluates CfC backbones against standard architectures across multiple continual learning strategies:

- **Datasets**: MNIST (5 tasks), CIFAR-10 (5 tasks), Tennessee Eastman Process (22 tasks)
- **Backbones**: Standard MLPs/CNNs vs. CfC variants
- **Models**: 10+ continual learning algorithms
- **Seeds**: 3 runs per configuration for statistical significance

## Quick Start

### 1. Run All Benchmarks (Full Paper Results)

```bash
cd /home/fneubuerger/CFC_Continual_Learning
./scripts/benchmarks/run_paper_benchmarks.sh --dataset all --max-parallel 4
```

**Estimated runtime**: 50-75 hours with 4 GPUs in parallel

### 2. Run Specific Dataset

```bash
# MNIST only (~2-3 hours per config × 30 configs = ~60-90 hours sequential)
./scripts/benchmarks/run_paper_benchmarks.sh --dataset mnist --max-parallel 4

# CIFAR-10 only (~8-12 hours per config × 20 configs = ~160-240 hours sequential)
./scripts/benchmarks/run_paper_benchmarks.sh --dataset cifar10 --max-parallel 4

# Tennessee Eastman Process only (~4-6 hours per config × 16 configs)
./scripts/benchmarks/run_paper_benchmarks.sh --dataset tep --max-parallel 4
```

### 3. Dry Run (Preview Commands)

```bash
./scripts/benchmarks/run_paper_benchmarks.sh --dataset mnist --dry-run
```

## Experiment Configuration

### MNIST Experiments (30 configurations × 3 seeds = 90 runs)

**MLP Baseline** (10 epochs):
- SGD (catastrophic forgetting baseline)
- Joint (upper bound - all data)
- ER (buffer: 200, 500)
- DER++ (buffer: 200, 500)
- ER-ACE (buffer: 200)
- A-GEM (buffer: 200)
- GEM (buffer: 200)
- EWC-Online
- SI (Synaptic Intelligence)
- LwF (Learning without Forgetting)

**CfC Backbone** (10 epochs):
- Same 10 models as MLP baseline

### CIFAR-10 Experiments (20 configurations × 3 seeds = 60 runs)

**ResNet-18 Baseline** (50 epochs):
- SGD
- Joint
- ER (buffer: 200, 500, 1000)
- DER++ (buffer: 200, 500, 1000)
- ER-ACE (buffer: 200, 500, 1000)
- EWC-Online
- SI

**CfC-CNN Backbone** (50 epochs):
- SGD
- Joint
- ER (buffer: 200, 500, 1000)
- DER++ (buffer: 200, 500, 1000)
- ER-ACE (buffer: 200, 500, 1000)
- EWC-Online

### Tennessee Eastman Process (16 configurations × 3 seeds = 48 runs)

**CfC Backbone** (20 epochs):
- SGD
- ER (buffer: 200, 500, 1000)
- DER++ (buffer: 200, 500, 1000)
- EWC-Online

**LSTM Baseline** (20 epochs):
- Same 4 models as CfC

## Monitoring Progress

### List Active Experiments
```bash
tmux ls
```

### Attach to Specific Experiment
```bash
tmux attach -t paper_mnist_cfc_er200_s0
```

### View Logs
```bash
tail -f paper_results/logs/mnist_cfc_er200_seed0.log
```

### Check Overall Progress
```bash
ls -lh paper_results/logs/*.log | wc -l  # Count completed/running
```

## Analyzing Results

### Generate Summary Tables

```bash
python scripts/analysis/analyze_paper_results.py \
    --results-dir paper_results \
    --output-dir paper_results/analysis
```

This produces:

1. **CSV Summaries**:
   - `summary_all.csv` - All experiments aggregated
   - `summary_mnist.csv` - MNIST-specific results
   - `summary_cifar.csv` - CIFAR-10-specific results
   - `summary_tep.csv` - TEP-specific results

2. **LaTeX Tables**:
   - `table_mnist.tex` - Ready for paper
   - `table_cifar.tex`
   - `table_tep.tex`

3. **Console Output**:
   - Best results per dataset
   - Completion statistics
   - Error summaries

### Example Output

```
BEST RESULTS (by Final Class-IL Accuracy)
--------------------------------------------------------------------------------

MNIST:
  backbone      model    buffer_size  final_class_il  final_task_il
  mnistcfc      joint    NaN          99.87           99.91
  mnistmlp      joint    NaN          99.82           99.88
  mnistcfc      derpp    500          98.45           98.72
  mnistmlp      derpp    500          97.89           98.21
  mnistcfc      er       500          97.23           97.56
```

## Key Metrics

Each experiment reports:

- **Class-IL Accuracy**: Average accuracy across all seen tasks (harder)
- **Task-IL Accuracy**: Accuracy when task identity is known (easier)
- **Forgetting**: Drop in accuracy on old tasks
- **Final Accuracy**: Performance after all tasks learned

## Directory Structure

```
paper_results/
├── logs/                           # Execution logs
│   ├── mnist_cfc_er200_seed0.log
│   ├── mnist_mlp_er200_seed0.log
│   └── ...
├── mnist_cfc_er200_seed0.csv       # Raw accuracy data
├── mnist_mlp_er200_seed0.csv
├── ...
└── analysis/                       # Generated summaries
    ├── raw_results.csv
    ├── summary_all.csv
    ├── summary_mnist.csv
    ├── table_mnist.tex
    └── ...
```

## Expected Results

Based on continual learning literature and CfC properties:

### MNIST
- **Upper Bound (Joint)**: ~99.8%
- **Good CL Performance**: >95% Class-IL
- **CfC Advantage**: Better temporal modeling → less forgetting

### CIFAR-10
- **Upper Bound (Joint)**: ~90-95%
- **Good CL Performance**: >70% Class-IL
- **CfC vs ResNet**: Competitive with fewer parameters

### Tennessee Eastman Process
- **Challenge**: 22 tasks with temporal sequences
- **CfC Advantage**: Explicitly designed for temporal data
- **Expected**: CfC > LSTM > standard methods

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size or buffer size
./scripts/benchmarks/run_paper_benchmarks.sh --dataset mnist
# Then manually edit batch_size in the script
```

### Experiment Crashed
```bash
# Check log for errors
tail -100 paper_results/logs/mnist_cfc_er200_seed0.log

# Restart specific experiment
tmux attach -t paper_mnist_cfc_er200_s0
# Press Ctrl+C, then up arrow, Enter to rerun
```

### Too Slow
```bash
# Increase parallelism (if you have GPUs)
./scripts/benchmarks/run_paper_benchmarks.sh --dataset mnist --max-parallel 8

# Or run subsets
./scripts/benchmarks/run_paper_benchmarks.sh --dataset mnist  # Fast (90 hours → 22 hours with 4 GPUs)
./scripts/benchmarks/run_paper_benchmarks.sh --dataset tep    # Medium (48 hours → 12 hours with 4 GPUs)
# Skip CIFAR-10 initially (slowest: 240 hours → 60 hours with 4 GPUs)
```

## Paper Workflow

1. **Run Benchmarks** (~3-7 days with 4 GPUs):
   ```bash
   ./scripts/benchmarks/run_paper_benchmarks.sh --dataset all --max-parallel 4
   ```

2. **Monitor Progress** (daily):
   ```bash
   tmux ls
   ls paper_results/logs/*.log | wc -l
   ```

3. **Generate Analysis** (once complete):
   ```bash
   python scripts/analysis/analyze_paper_results.py
   ```

4. **Create Visualizations**:
   ```bash
   python scripts/analysis/visualize_results.py
   ```

5. **Extract Tables for Paper**:
   ```bash
   # LaTeX tables are in paper_results/analysis/
   cat paper_results/analysis/table_mnist.tex
   ```

## Configuration Details

All hyperparameters are documented in:
- `configs/paper_benchmarks.yaml` - High-level configuration
- `scripts/benchmarks/run_paper_benchmarks.sh` - Execution details
- `mammoth/utils/best_args.py` - Mammoth's default hyperparameters

## Citation

If you use this benchmark suite, please cite:

```bibtex
@article{cfc-continual-learning,
  title={Closed-form Continuous-time Neural Networks for Continual Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

For questions or issues:
- Open an issue on GitHub
- Check logs in `paper_results/logs/`
- Review Mammoth documentation: https://github.com/aimagelab/mammoth
