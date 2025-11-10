# Quick Start Guide

## Setup Complete! ✅

Your CFC Continual Learning environment has been successfully configured.

## What's Been Done

1. **Fixed Critical Bugs** in CfC implementations
   - `mammoth/backbone/MNISTcfc.py` - Now properly processes sequential data
   - `mammoth/backbone/cnn_cfc.py` - CfC correctly integrated with CNN

2. **Created Documentation**
   - `docs/theoretical_foundations.md` - Hypotheses and theory
   - `SCIENTIFIC_IMPROVEMENT_PLAN.md` - 12-week research plan
   - `SUMMARY.md` - Complete project summary

3. **Set Up Environment**
   - Poetry configuration with all dependencies
   - Python 3.10 virtual environment
   - Analysis tools ready to use

4. **Analyzed Existing Data**
   - 11 WandB runs parsed
   - Visualizations generated in `analysis/results/`
   - Gaps identified (no CfC experiments yet!)

## Quick Commands

### Activate Environment
```bash
cd /home/fneubuerger/CFC_Continual_Learning
poetry shell
```

### Run Experiments

**Test Fixed Implementation:**
```bash
# Quick test that CfC architecture works
poetry run python -c "
from mammoth.backbone.MNISTcfc import MNISTcfc
import torch
model = MNISTcfc(784, 10, use_ncp_wiring=True)
x = torch.randn(16, 1, 28, 28)
out = model(x)
print(f'✓ MNISTcfc works! Output shape: {out.shape}')
"
```

**Run Baseline Experiments:**
```bash
# Sequential MNIST with different methods
poetry run python mammoth/utils/main.py --dataset seq-mnist --model sgd --seed 0
poetry run python mammoth/utils/main.py --dataset seq-mnist --model er --seed 0
poetry run python mammoth/utils/main.py --dataset seq-mnist --model ewc --seed 0
```

**Run Multiple Seeds:**
```bash
for seed in {0..4}; do
    poetry run python mammoth/utils/main.py --dataset seq-mnist --model er --seed $seed
done
```

### Analyze Results

```bash
# Generate analysis and visualizations
poetry run python analysis/wandb_analysis.py

# View results
ls analysis/results/
```

### Development

```bash
# Format code
poetry run black mammoth/ analysis/

# Run tests
poetry run pytest mammoth/tests/

# Launch Jupyter for analysis
poetry run jupyter notebook
```

## Next Steps

### 1. Verify Implementations (Today)

```bash
# Test MNISTcfc
cd mammoth
poetry run python -c "
from backbone.MNISTcfc import MNISTcfc
import torch

# Test with NCP wiring
model_ncp = MNISTcfc(784, 10, use_ncp_wiring=True, hidden_size=128)
print(f'NCP model params: {sum(p.numel() for p in model_ncp.parameters())}')

# Test without NCP wiring (fully connected)
model_fc = MNISTcfc(784, 10, use_ncp_wiring=False, hidden_size=128)
print(f'FC model params: {sum(p.numel() for p in model_fc.parameters())}')

# Forward pass
x = torch.randn(8, 1, 28, 28)
out_ncp = model_ncp(x)
out_fc = model_fc(x)
print(f'✓ Both models work! NCP output: {out_ncp.shape}, FC output: {out_fc.shape}')
"
```

### 2. Update Dataset to Use CfC (Tomorrow)

Edit `mammoth/datasets/seq_mnist.py`:

```python
@staticmethod
def get_backbone():
    # OLD: return MNISTMLP(28 * 28, ...)
    # NEW:
    from backbone.MNISTcfc import MNISTcfc
    return MNISTcfc(28 * 28, 
                    SequentialMNIST.N_TASKS * SequentialMNIST.N_CLASSES_PER_TASK,
                    use_ncp_wiring=True,
                    hidden_size=128)
```

### 3. Run First CfC Experiments (This Week)

```bash
# Experiment 1: CfC vs. MLP baseline
poetry run python mammoth/utils/main.py --dataset seq-mnist --model sgd --seed 0

# Experiment 2: CfC + Experience Replay
poetry run python mammoth/utils/main.py --dataset seq-mnist --model er --buffer_size 500 --seed 0

# Experiment 3: Multiple seeds for statistical validity
for seed in {0..9}; do
    poetry run python mammoth/utils/main.py --dataset seq-mnist --model er --seed $seed
done
```

### 4. Ablation Studies (Next 2 Weeks)

Create experiments comparing:
- NCP wiring vs. fully connected CfC
- CfC vs. LSTM/GRU
- Different hidden sizes
- Different sparsity levels

### 5. Analysis and Visualization (Ongoing)

Monitor experiments in WandB:
https://wandb.ai/fneubuerger/mammoth/

## File Structure

```
CFC_Continual_Learning/
├── mammoth/                      # Continual learning framework
│   ├── backbone/
│   │   ├── MNISTcfc.py          # ✅ FIXED - CfC for MNIST
│   │   ├── cnn_cfc.py           # ✅ FIXED - CfC for CNN
│   │   └── ...
│   ├── datasets/                 # Dataset loaders
│   ├── models/                   # CL algorithms (ER, EWC, etc.)
│   └── utils/                    # Training utilities
├── ncps/                         # Neural Circuit Policies library
├── docs/
│   └── theoretical_foundations.md  # ✅ Hypotheses and theory
├── analysis/
│   ├── wandb_analysis.py         # ✅ Analysis script
│   └── results/                  # ✅ Generated visualizations
├── pyproject.toml                # ✅ Poetry configuration
├── SCIENTIFIC_IMPROVEMENT_PLAN.md # ✅ 12-week plan
├── SUMMARY.md                    # ✅ Complete summary
└── QUICKSTART.md                 # ✅ This file
```

## Common Issues

### Issue: "Module not found"
**Solution:** Make sure you're in the Poetry environment
```bash
poetry shell
```

### Issue: "CUDA not available"
**Solution:** Check if PyTorch sees your GPU
```bash
poetry run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Issue: "WandB not logging"
**Solution:** Check your WandB login
```bash
poetry run wandb login
```

### Issue: Import errors from mammoth
**Solution:** Run from the project root, not inside mammoth/
```bash
cd /home/fneubuerger/CFC_Continual_Learning
poetry run python mammoth/utils/main.py ...
```

## Key Findings from Analysis

⚠️ **Critical Gaps Identified:**

1. **No CfC experiments yet** - All 11 runs use standard backbones
2. **No random seeds** - Cannot assess statistical significance  
3. **Limited baselines** - Missing SGD, EWC, LwF comparisons
4. **Limited datasets** - Mostly seq-cifar10, need more variety

**Your implementations are now fixed and ready to address these gaps!**

## Resources

- **WandB Dashboard:** https://wandb.ai/fneubuerger/mammoth/
- **NCP Paper:** https://www.nature.com/articles/s42256-022-00556-7
- **CfC Paper:** https://www.nature.com/articles/s42256-022-00556-7
- **Mammoth Framework:** See mammoth/README.md

## Questions?

Review these files:
- `SUMMARY.md` - Complete overview
- `SCIENTIFIC_IMPROVEMENT_PLAN.md` - Detailed research plan
- `docs/theoretical_foundations.md` - Why NCPs might help

## Ready to Go! 🚀

Your environment is set up and ready for experiments. Start with verifying the implementations work, then run your first CfC continual learning experiments!

Good luck with your research! 🎓
