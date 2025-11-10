# Benchmark Configuration Files

This directory contains YAML configuration files for running CfC continual learning benchmarks.

## Usage

Run benchmarks using the `benchmark_runner.py` script:

```bash
# Dry run to see what would be executed
python benchmark_runner.py --config configs/quick_test.yaml --dry-run

# Run actual experiments (sequential)
python benchmark_runner.py --config configs/quick_test.yaml

# Run experiments in parallel (2 GPUs)
python benchmark_runner.py --config configs/quick_test.yaml --parallel 2

# Run with more parallel processes
python benchmark_runner.py --config configs/tep_benchmark.yaml --parallel 4
```

## Configuration Files

### `quick_test.yaml`
- **Purpose**: Fast testing of the benchmark runner
- **Duration**: ~10-15 minutes
- **Experiments**: 2 (SGD + ER with mnistcfc on seq-mnist)
- **Use case**: Testing setup, debugging

### `cfc_only.yaml`
- **Purpose**: Compare CfC variants across methods
- **Duration**: ~8-12 hours
- **Experiments**: 30 (5 models × 2 CfC backbones × 2 datasets × 3 seeds)
- **Use case**: CfC architecture evaluation

### `tep_benchmark.yaml`
- **Purpose**: Tennessee Eastman Process benchmarks (Neural + Traditional ML)
- **Duration**: ~12-18 hours (parallel)
- **Experiments**: 42 (12 traditional ML + 30 neural networks)
- **Features**: Compares CfC against tree-based methods (XGBoost, LightGBM, Random Forest)
- **Use case**: TEP fault detection comparison

### `full_benchmark_parallel.yaml`
- **Purpose**: Comprehensive parallel benchmark across all datasets
- **Duration**: ~2-3 days (with --parallel 2)
- **Experiments**: ~200+ (includes traditional ML for TEP)
- **Features**: Full comparison with parallel execution
- **Use case**: Complete evaluation for paper

### `benchmark_config.yaml`
- **Purpose**: Comprehensive benchmark including baselines
- **Duration**: ~24-48 hours
- **Experiments**: 144 (6 models × 4 backbones × 3 datasets × 2 seeds)
- **Use case**: Full comparison with baselines

### `paper_experiments.yaml`
- **Purpose**: Publication-ready experiments with high statistical power
- **Duration**: ~3-5 days
- **Experiments**: 360 (6 models × 4 backbones × 3 datasets × 5 seeds)
- **Use case**: Final paper results

## Configuration Format

```yaml
# Python environment
python_cmd: python
venv_path: ../.venv/bin/activate
base_path: ./mammoth

# GPU configuration for parallel execution
gpus:
  - 0
  - 1

# Global arguments (applied to all experiments)
global_args:
  n_epochs: 10
  batch_size: 32
  num_workers: 0

# Experiment grid
datasets:
  - seq-mnist
  - seq-cifar10
  - seq-tep

models:
  - sgd
  - er

backbones:
  - mnistcfc
  - cnn-cfc
  - tepcfc

seeds:
  - 0
  - 42

# Model-specific arguments
model_args:
  er:
    buffer_size: 500
    minibatch_size: 32

# Backbone-specific arguments
backbone_args:
  mnistcfc:
    input_size: 784
    hidden_size: 256
    use_ncp_wiring: true
  
  tepcfc:
    input_size: 52
    output_size: 21
    hidden_size: 128
    use_ncp_wiring: true

# Dataset-specific arguments
dataset_args:
  seq-mnist:
    lr: 0.03
  
  seq-tep:
    lr: 0.001

# Traditional ML methods (tree-based)
traditional_ml:
  models:
    - xgboost
    - lightgbm
    - random_forest
  
  datasets:
    - seq-tep
  
  seeds:
    - 0
    - 42
  
  args:
    output_dir: tep_ml_results
  
  model_args:
    xgboost:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
```

## Output Structure

Results are saved to `benchmark_results/run_<timestamp>/`:

```
benchmark_results/
└── run_20251110_140530/
    ├── config.yaml                          # Copy of configuration
    ├── summary.json                         # Experiment summary
    ├── er_mnistcfc_seq-mnist_seed0_output.txt
    ├── sgd_mnistcfc_seq-mnist_seed0_output.txt
    └── ...
```

## Creating Custom Configurations

1. Copy an existing config file
2. Modify the experiment grid (datasets, models, backbones, seeds)
3. Adjust model/backbone/dataset-specific arguments
4. Run with `--dry-run` first to verify

## Example Workflows

### Quick validation test
```bash
python benchmark_runner.py --config configs/quick_test.yaml
```

### CfC architecture comparison
```bash
python benchmark_runner.py --config configs/cfc_only.yaml
```

### Full benchmark with baselines
```bash
# First do a dry run
python benchmark_runner.py --config configs/benchmark_config.yaml --dry-run

# Then run with parallel execution (use 2 GPUs)
nohup python benchmark_runner.py --config configs/benchmark_config.yaml --parallel 2 > benchmark.log 2>&1 &
```

### TEP benchmarks with traditional ML
```bash
# Run TEP benchmarks (neural + tree-based) in parallel
python benchmark_runner.py --config configs/tep_benchmark.yaml --parallel 2
```

### Paper experiments
```bash
# Run comprehensive parallel benchmarks
nohup python benchmark_runner.py --config configs/full_benchmark_parallel.yaml --parallel 2 > full_benchmark.log 2>&1 &
```

## Tips

- **Start small**: Use `quick_test.yaml` to verify everything works
- **Dry run**: Always use `--dry-run` first to check commands
- **Parallel execution**: Use `--parallel N` to run N experiments simultaneously
- **GPU assignment**: Specify GPUs in config file under `gpus: [0, 1]`
- **Background execution**: Use `nohup` for long-running experiments
- **Resource management**: Adjust `num_workers` based on available CPU/GPU
- **Checkpoint saving**: Use `savecheck: task` to save after each task
- **Statistical power**: Use at least 3-5 seeds for publication results
- **Traditional ML**: Use `traditional_ml` section for tree-based methods (XGBoost, LightGBM, etc.)
- **Mixed experiments**: Configs can include both neural networks and traditional ML methods

## Monitoring Progress

Check experiment progress:
```bash
# View live output
tail -f benchmark_results/run_*/summary.json

# Count completed experiments
ls benchmark_results/run_*/*_output.txt | wc -l
```
