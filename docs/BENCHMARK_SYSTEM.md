# Benchmark Runner System - Overview

## What We Built

A comprehensive, configuration-driven benchmark system for running CfC continual learning experiments with:

1. **Parallel execution** - Run multiple experiments simultaneously across GPUs
2. **Neural networks** - All Mammoth continual learning methods (ER, DER++, LwF, EWC, iCaRL, etc.)
3. **Traditional ML** - Tree-based methods (XGBoost, LightGBM, Random Forest, Gradient Boosting)
4. **Multiple datasets** - MNIST, CIFAR-10, CIFAR-100, Tennessee Eastman Process
5. **CfC architectures** - All variants (mnistcfc, cnn-cfc, tepcfc, teplstm)
6. **Statistical rigor** - Multiple random seeds for reproducibility

## Architecture

```
benchmark_runner.py
├── Reads YAML config files
├── Generates experiment grid (datasets × models × backbones × seeds)
├── Handles both neural networks and traditional ML
├── Executes in parallel with GPU assignment
└── Saves results + logs

configs/
├── quick_test.yaml              # Fast testing (2 experiments, ~15 min)
├── cfc_only.yaml                # CfC comparison (30 experiments, ~8-12 hrs)
├── tep_benchmark.yaml           # TEP + traditional ML (42 experiments, ~12-18 hrs)
├── full_benchmark_parallel.yaml # Everything (200+ experiments, ~2-3 days)
├── benchmark_config.yaml        # Full comparison (144 experiments)
└── paper_experiments.yaml       # Publication-ready (360 experiments)
```

## Key Features

### 1. Parallel Execution
```bash
# Run 2 experiments simultaneously on 2 GPUs
python benchmark_runner.py --config configs/tep_benchmark.yaml --parallel 2
```

Round-robin GPU assignment:
- Experiment 1 → GPU 0
- Experiment 2 → GPU 1
- Experiment 3 → GPU 0
- Experiment 4 → GPU 1
- ...

### 2. Traditional ML Support
Automatically detects and runs tree-based methods:
```yaml
traditional_ml:
  models:
    - xgboost
    - lightgbm
    - random_forest
  datasets:
    - seq-tep
  model_args:
    xgboost:
      n_estimators: 100
      max_depth: 6
```

### 3. Flexible Configuration
Hierarchical argument merging:
```
global_args → dataset_args → model_args → backbone_args
```

Example:
- `global_args`: `n_epochs: 10` (applies to all)
- `dataset_args[seq-mnist]`: `lr: 0.03` (MNIST-specific)
- `model_args[er]`: `buffer_size: 500` (ER-specific)
- `backbone_args[mnistcfc]`: `hidden_size: 256` (CfC-specific)

### 4. Experiment Grid Generation
Automatically creates all combinations:
```yaml
datasets: [seq-mnist, seq-cifar10, seq-tep]
models: [sgd, er, derpp]
backbones: [mnistcfc, cnn-cfc, tepcfc]
seeds: [0, 42, 123]
```
= 3 × 3 × 3 × 3 = **81 experiments**

### 5. Results Organization
```
benchmark_results/
└── run_20251110_140530/
    ├── config.yaml                          # Reproducibility
    ├── summary.json                         # Metadata + results
    ├── er_mnistcfc_seq-mnist_seed0_output.txt
    ├── er_mnistcfc_seq-mnist_seed42_output.txt
    ├── xgboost_seq-tep_seed0_output.txt
    └── ...
```

## Example Workflows

### Quick Test (Verify Setup)
```bash
# Dry run
python benchmark_runner.py --config configs/quick_test.yaml --dry-run

# Real run
python benchmark_runner.py --config configs/quick_test.yaml
```
**Output**: 2 experiments, ~15 minutes

### CfC Comparison
```bash
python benchmark_runner.py --config configs/cfc_only.yaml --parallel 2
```
**Output**: 30 experiments (5 methods × 2 CfC backbones × 2 datasets × 3 seeds)

### TEP Benchmarks (Neural + Traditional ML)
```bash
python benchmark_runner.py --config configs/tep_benchmark.yaml --parallel 2
```
**Output**:
- 12 traditional ML experiments (4 models × 3 seeds)
- 30 neural network experiments (5 methods × 2 backbones × 3 seeds)
- **Total: 42 experiments**

### Full Parallel Benchmark
```bash
nohup python benchmark_runner.py --config configs/full_benchmark_parallel.yaml --parallel 2 > benchmark.log 2>&1 &
```
**Output**: 200+ experiments across all datasets, methods, and backbones

## Configuration Examples

### Minimal Config
```yaml
python_cmd: python
venv_path: ../.venv/bin/activate
base_path: ./mammoth

datasets: [seq-mnist]
models: [sgd]
backbones: [mnistcfc]
seeds: [0]

global_args:
  n_epochs: 5
  batch_size: 32
```

### With GPU Assignment
```yaml
gpus: [0, 1]  # Use GPUs 0 and 1
```

### With Traditional ML
```yaml
traditional_ml:
  models: [xgboost, lightgbm]
  datasets: [seq-tep]
  seeds: [0, 42]
  args:
    output_dir: ml_results
  model_args:
    xgboost:
      n_estimators: 100
```

### With Hierarchical Args
```yaml
global_args:
  n_epochs: 10
  batch_size: 32

model_args:
  er:
    buffer_size: 500
  derpp:
    buffer_size: 500
    alpha: 0.1

backbone_args:
  mnistcfc:
    hidden_size: 256
    use_ncp_wiring: true

dataset_args:
  seq-mnist:
    lr: 0.03
  seq-tep:
    lr: 0.001
```

## Command Reference

```bash
# Basic usage
python benchmark_runner.py --config <config.yaml>

# Dry run (preview commands)
python benchmark_runner.py --config <config.yaml> --dry-run

# Parallel execution
python benchmark_runner.py --config <config.yaml> --parallel <N>

# Background execution
nohup python benchmark_runner.py --config <config.yaml> --parallel 2 > output.log 2>&1 &

# Monitor progress
tail -f output.log

# Check results
ls -lh benchmark_results/run_*/
cat benchmark_results/run_*/summary.json
```

## Integration with Existing Tools

The benchmark runner integrates with:

1. **Mammoth v2.0** - Main continual learning framework
2. **tep_gradient_boosting.py** - Traditional ML for TEP
3. **interpretability_analysis.py** - Post-hoc analysis
4. **benchmark_v2_replay.py** - Specialized replay method comparison

## For Your Paper

Recommended workflow:

1. **Quick validation**: Run `quick_test.yaml` to verify setup
2. **CfC evaluation**: Run `cfc_only.yaml` for CfC architecture comparison
3. **TEP comparison**: Run `tep_benchmark.yaml` for neural vs traditional ML
4. **Full results**: Run `full_benchmark_parallel.yaml` for complete paper results
5. **Interpretability**: Use `interpretability_analysis.py` on best checkpoints

## Benefits

- **Reproducible** - All configs saved with results  
- **Scalable** - Parallel execution across GPUs  
- **Flexible** - Easy to add datasets/models/backbones  
- **Comprehensive** - Neural + traditional ML in one system  
- **Organized** - Structured results with metadata  
- **Safe** - Dry-run mode to preview before execution  

## Next Steps

1. Run quick test to verify everything works
2. Customize configs for your specific experiments
3. Run small-scale experiments first (2-3 seeds)
4. Scale up to full paper experiments (5+ seeds)
5. Analyze results using provided tools
