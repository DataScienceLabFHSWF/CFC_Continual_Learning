# Integration Guide: Using Advanced Metrics in Mammoth

This document explains how to integrate tau monitoring and advanced metrics into the Mammoth training loop.

## 1. Tau Monitoring (for LTC backbones)

### In your training script:

```python
from utils.tau_monitor import get_tau_monitor

# Initialize monitor
tau_monitor = get_tau_monitor(enabled=True, log_every_n_steps=100)

# During training loop
for epoch in range(n_epochs):
    for task_id in range(n_tasks):
        # ... training code ...
        
        # Update tau monitoring
        tau_monitor.update(model.net, task_id, epoch, use_wandb=True)
    
    # At task boundary
    tau_monitor.on_task_end(task_id)

# After all tasks
stability_metrics = tau_monitor.analyze_stability()
print(stability_metrics)
```

## 2. Advanced Metrics

### In your training script:

```python
from utils.advanced_metrics import AdvancedMetricsManager

# Initialize
config = {
    'representational_stability': {'enabled': True},
    'weight_change': {'enabled': True},
    'gradient_interference': {'enabled': True}
}
metrics_manager = AdvancedMetricsManager(config)

# During training
for task_id in range(n_tasks):
    # Start of task
    metrics_manager.on_task_start(model.net, task_id, train_loader, device)
    
    for epoch in range(n_epochs):
        for batch in train_loader:
            # Forward
            outputs = model(batch)
            loss = criterion(outputs, targets)
            
            # Backward
            loss.backward()
            
            # Cache gradients (for interference analysis)
            metrics_manager.on_backward(model.net, task_id)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
    
    # End of task
    metrics_manager.on_task_end(model.net, task_id, train_loader, device)

# Final analysis
task_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]  # For 5-task scenario
metrics_manager.analyze_all(task_pairs)
```

## 3. Integration Points in Mammoth

### Modify `mammoth/utils/training.py`:

Add hooks in the main training function:

```python
def train(model, dataset, args):
    # Initialize metrics
    tau_monitor = None
    metrics_manager = None
    
    if args.backbone in ['mnistltc', 'tepltc', 'cnn-ltc']:
        tau_monitor = get_tau_monitor(enabled=True)
    
    if args.enable_advanced_metrics:
        metrics_manager = AdvancedMetricsManager(args.metrics_config)
    
    # Training loop
    for t in range(dataset.N_TASKS):
        if metrics_manager:
            metrics_manager.on_task_start(model.net, t, train_loader, args.device)
        
        # ... existing training code ...
        
        if tau_monitor:
            tau_monitor.update(model.net, t, epoch)
        
        if metrics_manager:
            # After backward pass
            metrics_manager.on_backward(model.net, t)
        
        # ... optimizer step ...
        
        if metrics_manager:
            metrics_manager.on_task_end(model.net, t, train_loader, args.device)
    
    # Final analysis
    if metrics_manager:
        task_pairs = [(i, i+1) for i in range(dataset.N_TASKS - 1)]
        metrics_manager.analyze_all(task_pairs)
```

## 4. Command Line Arguments

Add to `mammoth/utils/args.py`:

```python
# Tau monitoring
parser.add_argument('--enable-tau-monitoring', type=int, default=0,
                   help='Enable tau monitoring for LTC backbones')
parser.add_argument('--tau-log-frequency', type=int, default=100,
                   help='Log tau every N steps')

# Advanced metrics
parser.add_argument('--enable-advanced-metrics', type=int, default=0,
                   help='Enable advanced CL metrics')
parser.add_argument('--repr-stability', type=int, default=1,
                   help='Enable representational stability metric')
parser.add_argument('--weight-change', type=int, default=1,
                   help='Enable weight change analysis')
parser.add_argument('--gradient-interference', type=int, default=1,
                   help='Enable gradient interference analysis')
```

## 5. Usage Example

```bash
# Run MNIST LTC with all metrics
python utils/main.py \
    --dataset seq-mnist \
    --model er \
    --backbone mnistltc \
    --buffer_size 200 \
    --n_epochs 10 \
    --enable-tau-monitoring 1 \
    --enable-advanced-metrics 1 \
    --wandb_project cfc-continual-learning

# Run ablation: CfC vs Random Sparse (no tau monitoring)
python utils/main.py \
    --dataset seq-mnist \
    --model er \
    --backbone mnist_random_sparse \
    --buffer_size 200 \
    --n_epochs 10 \
    --enable-advanced-metrics 1 \
    --gradient-interference 1
```

## 6. Expected WandB Metrics

After integration, WandB will log:

### Tau Monitoring (LTC only):
- `tau_mean`, `tau_std`, `tau_min`, `tau_max`
- `tau_bimodality` (>0.555 = bimodal distribution)
- `tau_fast_count`, `tau_slow_count`
- `tau_distribution` (histogram)
- `tau_stability_X_to_Y` (correlation between tasks)

### Representational Stability:
- `repr_cosine_sim_mean_taskX` (closer to 1 = stable)
- `repr_l2_distance_mean_taskX` (closer to 0 = stable)
- `repr_relative_change_mean_taskX`

### Weight Change:
- `weight_change_frobenius_LAYER_tX_to_tY`
- `weight_change_relative_LAYER_tX_to_tY`

### Gradient Interference:
- `gradient_similarity_tX_tY` (closer to 0 = less interference)

## 7. Analysis Scripts

Create `scripts/analysis/analyze_metrics.py`:

```python
import wandb
import pandas as pd
import matplotlib.pyplot as plt

# Download metrics from WandB
api = wandb.Api()
runs = api.runs("your-entity/cfc-continual-learning")

# Compare tau distributions
for run in runs:
    if 'ltc' in run.config.get('backbone', ''):
        history = run.history()
        plt.hist(history['tau_distribution'], alpha=0.5, label=run.name)

plt.xlabel('Tau Value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('tau_comparison.png')
```

## 8. Testing

Test the integration:

```bash
# Quick test with 1 task
python utils/main.py \
    --dataset seq-mnist \
    --model sgd \
    --backbone mnistltc \
    --n_epochs 1 \
    --enable-tau-monitoring 1 \
    --enable-advanced-metrics 1

# Check that metrics are logged
# Expected output:
# [Tau Monitor] Task 0, Epoch 0
#   Mean: 5.2341 ± 2.1234
#   Bimodality: 0.612 (bimodal)
# [Repr Stability] Task 0:
#   repr_cosine_sim_mean: 0.9823
```
