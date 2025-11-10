# Mammoth v2.0 Benchmarking Guide

This document describes the updated benchmarking scripts for Mammoth v2.0.

## New Scripts for v2.0

### 1. Replay Methods Benchmark (`benchmark_v2_replay.py`)

Parallel benchmarking of replay-based continual learning methods.

**Usage:**
```bash
# Run with default backbone (mnistmlp)
python benchmark_v2_replay.py

# Run with CfC backbone
python benchmark_v2_replay.py --backbone mnistcfc --seeds 3

# Custom output path
python benchmark_v2_replay.py --output results/v2_cfc_replay.json
```

**Methods Tested:**
- Experience Replay (ER)
- Dark Experience Replay (DER, DER++)
- GDumb
- GSS (Gradient-based Sample Selection)
- HAL (Hindsight Anchor Learning)
- iCaRL
- MER (Meta-Experience Replay)
- ER-ACE
- X-DER variants
- FDR (Feature Distillation Replay)

**GPU Configuration:**
- Runs on 2 GPUs (configurable via `NUM_GPUS`)
- 3 experiments per GPU in parallel
- Automatic distribution via `CUDA_VISIBLE_DEVICES`

### 2. Interpretability Analysis (`interpretability_analysis.py`)

Extract and visualize CfC network interpretability for continual learning.

**Usage:**
```bash
# Analyze a trained model
python interpretability_analysis.py \
  --checkpoint checkpoints/seq-mnist-mnistcfc.pt \
  --dataset seq-mnist \
  --backbone mnistcfc \
  --output_dir interpretability_results
```

**Outputs:**
- `report.json` - Quantitative interpretability metrics
- `figures/cfc_wiring.pdf` - NCP wiring structure visualization
- `figures/activation_patterns.pdf` - Neuron activation across tasks
- `figures/critical_pathways.pdf` - Task-critical neuron identification

**Metrics:**
- Neuron specialization score
- Task-critical pathway identification
- Neuron activation patterns
- Wiring sparsity analysis

### 3. Gradient Boosting Baseline (`tep_gradient_boosting.py`)

Traditional ML baseline for Tennessee Eastman Process fault detection.

**Usage:**
```bash
# Run XGBoost and LightGBM
python tep_gradient_boosting.py \
  --data_dir data/tennessee_eastman \
  --models xgboost lightgbm \
  --output results/tep_gb_baseline.json
```

**Installation:**
```bash
pip install xgboost lightgbm
```

**Outputs:**
- Overall accuracy comparison
- Per-fault detection rates
- F1 scores (macro/weighted)
- Training time comparison
- Feature importance analysis

## API Changes from v1.x to v2.0

### Command Line Arguments

**v1.x:**
```bash
python utils/main.py --dataset seq-mnist --model er \
  --input_size 784 --output_size 10 --kwargs "{}" \
  --nowand 1
```

**v2.0:**
```bash
python utils/main.py --dataset seq-mnist --model er \
  --backbone mnistmlp --input_size 784 --output_size 10
# No --nowand needed (just don't set wandb_entity/wandb_project)
```

### Key Changes

1. **Backbone Arguments:**
   - v1.x: `--input_size`, `--output_size`, `--kwargs`
   - v2.0: `--backbone <name>` + automatic parameter parsing

2. **Wandb Control:**
   - v1.x: `--nowand 1`
   - v2.0: Automatically disabled if `wandb_entity` and `wandb_project` not set

3. **Backbone Registration:**
   ```python
   # v1.x
   class MNISTcfc(nn.Module):
       pass
   
   # v2.0
   @register_backbone('mnistcfc')
   def mnistcfc(input_size: int = 784, output_size: int = 10, ...):
       return BaseMNISTcfc(input_size, output_size, ...)
   ```

4. **Forward Method:**
   ```python
   # v1.x
   def forward(self, x, return_features=False):
       ...
   
   # v2.0
   def forward(self, x, returnt='out'):
       if returnt == 'out': return logits
       elif returnt == 'features': return features
       elif returnt in ['both', 'all']: return logits, features
   ```

## Running Benchmarks

### Quick Test (1 method, 1 seed)
```bash
cd mammoth
python utils/main.py --dataset seq-mnist --model sgd \
  --backbone mnistcfc --n_epochs 1 --batch_size 32 \
  --lr 0.1 --input_size 784 --output_size 10
```

### Full Replay Benchmark (13 methods × 3 seeds)
```bash
python benchmark_v2_replay.py --backbone mnistcfc --seeds 3
# Takes ~2-3 hours on 2x H200 GPUs
```

### Interpretability Analysis
```bash
# 1. Train a model
cd mammoth
python utils/main.py --dataset seq-mnist --model er \
  --backbone mnistcfc --n_epochs 5 --buffer_size 500 \
  --savecheck task --ckpt_name er_mnistcfc

# 2. Analyze interpretability
cd ..
python interpretability_analysis.py \
  --checkpoint mammoth/checkpoints/er_mnistcfc.pt \
  --dataset seq-mnist --backbone mnistcfc
```

### TEP Gradient Boosting Baseline
```bash
# Install dependencies
pip install xgboost lightgbm

# Run baseline
python tep_gradient_boosting.py --data_dir data/tennessee_eastman
```

## Results Organization

```
results/
├── v2_replay_benchmark.json          # Replay methods results
├── v2_cfc_replay.json                # CfC backbone results
├── tep_gb_baseline.json              # Gradient boosting baseline
└── interpretability_results/
    ├── report.json                   # Quantitative metrics
    └── figures/
        ├── cfc_wiring.pdf
        ├── activation_patterns.pdf
        └── critical_pathways.pdf
```

## Performance Comparison

### Expected Results (seq-mnist, Class-IL)

| Method | MLP Backbone | CfC Backbone | Difference |
|--------|-------------|--------------|------------|
| SGD | 19% | 22% | +3% |
| ER | 54% | 58% | +4% |
| DER++ | 72% | 75% | +3% |
| iCaRL | 65% | 68% | +3% |

*Note: CfC shows consistent improvement due to temporal dynamics*

### TEP Fault Detection

| Method | Accuracy | F1 Macro | Notes |
|--------|----------|----------|-------|
| XGBoost | ~85% | ~82% | Baseline (no temporal) |
| LightGBM | ~86% | ~83% | Slightly better |
| CfC (Joint) | ~95% | ~94% | Upper bound |
| CfC (Incremental) | ~88% | ~85% | With continual learning |

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'xitorch'"
**Solution:** These are warnings for optional models. Your method still runs fine.

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size or number of parallel experiments:
```python
# In benchmark script
NUM_GPUS = 2
EXPERIMENTS_PER_GPU = 2  # Reduce from 3 to 2
```

### Issue: Benchmark scripts using old API
**Solution:** Use the new `benchmark_v2_*.py` scripts instead of old ones.

## Next Steps

1. **Complete v2.0 Migration:**
   - Run old-mammoth benchmarks for baseline
   - Compare v1.x vs v2.0 results
   - Merge to main branch

2. **Interpretability Paper:**
   - Generate interpretability results for all CfC models
   - Analyze neuron specialization across tasks
   - Create visualization for publication

3. **TEP Continual Learning:**
   - Compare GB baseline vs CfC methods
   - Implement incremental learning scenarios
   - Measure forgetting on industrial data

## References

- Mammoth v2.0: [https://github.com/SequelONE/mammoth](https://github.com/SequelONE/mammoth)
- CfC Networks: Hasani et al., "Closed-form Continuous-time Neural Networks", Nature MI 2022
- TEP Dataset: Downs & Vogel, "A Plant-Wide Industrial Process Control Problem", 1993
