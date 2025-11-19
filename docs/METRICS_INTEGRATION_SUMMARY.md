# Metrics Integration Summary

## Overview
Successfully integrated advanced metrics and tau monitoring into the Mammoth training loop.

## What Was Changed

### 1. Command-Line Arguments (`mammoth/utils/args.py`)
Added three new arguments to the Management group:
- `--enable_advanced_metrics`: Enable Representational Stability, Weight Change, and Gradient Interference metrics
- `--enable_tau_monitor`: Enable tau (time constant) monitoring for LTC-based backbones
- `--tau_log_interval`: Set logging frequency for tau statistics (default: 100 steps)

### 2. Training Loop (`mammoth/utils/training.py`)

#### Imports
```python
from utils.tau_monitor import get_tau_monitor
from utils.advanced_metrics import AdvancedMetricsManager
```

#### Initialization (before training starts)
- Tau monitor: Initialized if `--enable_tau_monitor=1` and backbone contains "ltc"
- Metrics manager: Initialized if `--enable_advanced_metrics=1`

#### Training Hooks

**Task Start (`model.meta_begin_task`)**
```python
metrics_manager.on_task_start(model.net, cur_task, train_loader, device)
```
- Captures initial model state
- Stores initial representations for stability analysis

**During Training (`train_single_epoch`)**
```python
# After backward pass
metrics_manager.on_backward(model.net, cur_task)  # Cache gradients
tau_monitor.update(model.net, cur_task, epoch, use_wandb=True)  # Log tau stats
```
- Monitors gradient interference between tasks
- Tracks tau distribution evolution

**Task End (`model.meta_end_task`)**
```python
metrics_manager.on_task_end(model.net, cur_task, train_loader, device)
tau_monitor.on_task_end(cur_task)
```
- Computes representational stability
- Measures weight changes
- Finalizes tau statistics for the task

**Final Analysis (after all tasks)**
```python
tau_monitor.analyze_stability()  # Tau stability metrics
metrics_manager.analyze_all(task_pairs)  # Cross-task interference
```
- Aggregates metrics across all tasks
- Logs to WandB if enabled

### 3. Configuration (`configs/ablation_benchmarks.yaml`)
Added global settings:
```yaml
global_settings:
  enable_other_metrics: 1
  enable_advanced_metrics: 1
  enable_tau_monitor: 1
  tau_log_interval: 100
  nowand: 0
```

### 4. Testing

#### Unit Test (`tests/test_metrics_integration.py`)
Tests:
- ✓ Argument parsing
- ✓ Module imports
- ✓ Tau monitor initialization
- ✓ Metrics manager initialization
- ✓ Training function signature

#### End-to-End Test (`tests/validate_metrics_e2e.py`)
Minimal MNIST experiment with:
- 2 tasks, 1 epoch each
- Advanced metrics enabled
- Tau monitoring enabled
- Validates full integration

## How to Use

### Run experiments with metrics enabled:
```bash
python utils/main.py \
  --dataset seq-mnist \
  --model sgd \
  --backbone mnistltc \
  --lr 0.03 \
  --n_epochs 10 \
  --batch_size 32 \
  --enable_advanced_metrics 1 \
  --enable_tau_monitor 1 \
  --tau_log_interval 100
```

### Run ablation benchmarks (includes metrics by default):
```bash
python utils/main.py --config ../configs/ablation_benchmarks.yaml
```

## Metrics Output

### WandB Logging
When `--nowand=0`:
- `tau_mean`, `tau_std`, `tau_bimodality_score` (every N steps)
- `tau_stability` (final analysis)
- `representational_stability`, `weight_change`, `gradient_interference` (per task)
- `advanced_metrics` (final summary)

### Console Logging
```
Tau Stability Analysis: {...}
Advanced Metrics Summary: {...}
```

## Error Handling
All metrics operations are wrapped in try-except blocks:
- Failed metric operations log warnings but don't crash training
- Graceful degradation if modules are missing

## Backward Compatibility
- Metrics are **opt-in** (default: disabled)
- No changes to existing experiments unless flags are explicitly set
- Works with all existing backbones (tau monitor only activates for LTC)

## Next Steps
1. ✓ Integration complete
2. Run validation test: `python tests/validate_metrics_e2e.py`
3. Launch ablation experiments: See `QUICK_START_BENCHMARKS.sh`
