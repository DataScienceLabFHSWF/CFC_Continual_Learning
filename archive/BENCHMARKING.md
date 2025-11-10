# Benchmarking Guide

This project includes comprehensive benchmarking scripts to evaluate continual learning methods across multiple datasets in parallel.

## Hardware Requirements

The benchmarking scripts are optimized for multi-GPU systems. Current configuration:
- **2x NVIDIA H200 NVL GPUs** (144GB each)
- **6 parallel experiments** (3 per GPU)
- Automatic GPU assignment via `CUDA_VISIBLE_DEVICES`

## Quick Start

```bash
# Run comprehensive benchmark (all methods × all datasets)
./run_benchmarks.sh all

# Run only replay-based methods
./run_benchmarks.sh replay

# Run only regularization methods
./run_benchmarks.sh regularization

# Quick test (SGD + ER on seq-mnist, 1 seed)
./run_benchmarks.sh quick

# Custom benchmark
./run_benchmarks.sh custom --methods sgd er ewc_on --datasets seq-mnist --seeds 5
```

## Benchmark Scripts

### 1. `benchmark_replay_methods.py`
Tests replay-based continual learning methods:
- **Methods**: ER, DER, DER++, GDumb, GSS, HAL, iCaRL, MER, ER-ACE, X-DER, FDR
- **Buffer sizes**: 500 (2000 for iCaRL)
- **Seeds**: 3 per method
- **Output**: `results/replay_methods_benchmark.json`

### 2. `benchmark_regularization_methods.py`
Tests regularization-based methods:
- **Methods**: EWC-Online, SI, LwF, LwF-MC
- **Seeds**: 3 per method
- **Output**: `results/regularization_methods_benchmark.json`

### 3. `benchmark_all_methods.py`
Comprehensive benchmark across all methods and datasets:
- **Methods**: 19 methods (baselines, replay, regularization, architecture, distillation)
- **Datasets**: seq-mnist, perm-mnist, rot-mnist
- **Configurable**: Methods, datasets, seeds via command-line arguments
- **Output**: `results/benchmark_YYYYMMDD_HHMMSS.json`

## Configuration

### Parallel Execution

Edit the scripts to adjust parallelism:

```python
NUM_GPUS = 2                    # Number of GPUs
EXPERIMENTS_PER_GPU = 3         # Parallel experiments per GPU
```

**Total parallelism**: 6 experiments running simultaneously

### Method Configurations

Each method uses tuned hyperparameters in `METHOD_CONFIGS`:

```python
METHOD_CONFIGS = {
    'er': {'buffer_size': 500},
    'ewc_on': {'e_lambda': 1000, 'gamma': 1.0},
    'si': {'c': 0.5, 'xi': 1.0},
    # ... etc
}
```

### Dataset Configurations

```python
DATASET_CONFIGS = {
    'seq-mnist': {'lr': 0.03, 'n_epochs': 3, 'batch_size': 32},
    'perm-mnist': {'lr': 0.03, 'n_epochs': 3, 'batch_size': 32},
    'rot-mnist': {'lr': 0.03, 'n_epochs': 3, 'batch_size': 32},
}
```

## Output Format

### Console Output

```
Method          Avg Accuracy    Std       Avg Forgetting  Success   
--------------------------------------------------------------------------------
er              85.23%          1.45%     12.34%          3/3
ewc_on          82.45%          2.10%     15.67%          3/3
sgd             54.32%          3.20%     45.23%          3/3
```

### JSON Output

```json
[
  {
    "method": "er",
    "dataset": "seq-mnist",
    "seed": 0,
    "gpu_id": 0,
    "accuracy": 85.23,
    "forgetting": 12.34,
    "elapsed": 245.6,
    "success": true,
    "method_params": {"buffer_size": 500},
    "dataset_params": {"lr": 0.03, "n_epochs": 3}
  }
]
```

## Example Usage

### Run specific methods on multiple datasets

```bash
python benchmark_all_methods.py \
  --methods sgd er der ewc_on \
  --datasets seq-mnist perm-mnist rot-mnist \
  --seeds 5
```

### Run with different GPU configuration

Edit the script:
```python
NUM_GPUS = 4                    # Use 4 GPUs
EXPERIMENTS_PER_GPU = 2         # 2 experiments per GPU
```

### Test a single method

```bash
python benchmark_all_methods.py \
  --methods er \
  --datasets seq-mnist \
  --seeds 1
```

## Monitoring

### GPU Utilization

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Or use nvtop
nvtop
```

### Progress Tracking

The scripts print progress updates:
```
[GPU 0] Starting er_seq-mnist_s0
[GPU 1] Starting der_seq-mnist_s0
[GPU 0] ✓ er_seq-mnist_s0 - 245.6s - Acc: 85.23
Progress: 10/90 experiments completed
```

## Performance Estimates

With 2x H200 NVL GPUs (6 parallel experiments):

| Benchmark Type | Methods | Datasets | Seeds | Est. Time |
|---------------|---------|----------|-------|-----------|
| Replay only | 13 | 1 | 3 | ~30 min |
| Regularization | 4 | 1 | 3 | ~10 min |
| All methods | 19 | 3 | 3 | ~2-3 hours |
| Quick test | 2 | 1 | 1 | ~5 min |

*Times are wall-clock estimates based on parallel execution*

## Troubleshooting

### Out of Memory

Reduce parallel experiments:
```python
EXPERIMENTS_PER_GPU = 2  # Instead of 3
```

Or reduce batch size in `DATASET_CONFIGS`.

### Timeouts

Increase timeout (default 2 hours):
```python
timeout=7200  # 2 hours in seconds
```

### Failed Experiments

Check the JSON output for error details:
```python
import json
results = json.load(open('results/benchmark_*.json'))
failures = [r for r in results if not r['success']]
for f in failures:
    print(f"{f['method']}: {f.get('stderr', 'Unknown error')}")
```

## Next Steps

After benchmarking:
1. Analyze results with visualization scripts (TODO)
2. Select best-performing methods for TEP experiments
3. Run Bayesian CL implementations on promising baselines
4. Compare CfC vs LSTM backbones systematically
