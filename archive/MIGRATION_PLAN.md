# Mammoth v2.0 Migration Plan

## Overview
Migrate from customized Mammoth v1.x (25 methods) to Mammoth v2.0 (70+ methods) while preserving all CfC customizations.

## Strategy: Parallel Branches

### Branch Structure
```
main (current state)
├── old-mammoth (v1.x + CfC for benchmarking)
└── mammoth-v2-migration (upgrade branch)
    └── main (after successful migration)
```

## Phase 1: Branch Setup ✅

### Step 1.1: Commit Current Work
```bash
git add .
git commit -m "Pre-migration: All CfC customizations and benchmarking scripts"
```

### Step 1.2: Create Branches
```bash
# Create old-mammoth branch (frozen v1.x for benchmarks)
git checkout -b old-mammoth
git push -u origin old-mammoth

# Create migration branch from main
git checkout main
git checkout -b mammoth-v2-migration
```

## Phase 2: Backup Current Mammoth

### Step 2.1: Archive Old Mammoth
```bash
cd /home/fneubuerger/CFC_Continual_Learning
cp -r mammoth mammoth_v1_backup
```

### Step 2.2: Document Custom Files
**CfC Backbones:**
- `mammoth/backbone/MNISTcfc.py`
- `mammoth/backbone/cnn_cfc.py`
- `mammoth/backbone/TEPcfc.py`

**Custom Datasets:**
- `mammoth/datasets/tennessee_eastman.py`
- `mammoth/datasets/seq_mnist.py` (modified for MNISTcfc)
- `mammoth/datasets/perm_mnist.py` (modified for MNISTcfc)
- `mammoth/datasets/rot_mnist.py` (modified for MNISTcfc)

**Tests:**
- `tests/test_mnistcfc.py`
- `tests/test_cnn_cfc.py`
- `tests/test_tep_*.py`

## Phase 3: Install Mammoth v2.0

### Step 3.1: Remove Old Mammoth
```bash
rm -rf mammoth
```

### Step 3.2: Clone Mammoth v2.0
```bash
git clone https://github.com/aimagelab/mammoth.git
cd mammoth
git log -1  # Check version
```

### Step 3.3: Analyze v2.0 Structure
```bash
# Check what changed
ls -la
cat README.md
ls backbone/
ls datasets/
ls models/
```

## Phase 4: Port CfC Backbones

### Step 4.1: MNISTcfc.py
**Source:** `mammoth_v1_backup/backbone/MNISTcfc.py`
**Target:** `mammoth/backbone/MNISTcfc.py`

**Required Changes:**
- Check import paths (utils/conf.py structure)
- Verify mammoth_backbones compatibility
- Update registration if needed

### Step 4.2: cnn_cfc.py
**Source:** `mammoth_v1_backup/backbone/cnn_cfc.py`
**Target:** `mammoth/backbone/cnn_cfc.py`

**Required Changes:**
- ResNet18 import path
- Feature dimension handling
- Check if backbone interface changed

### Step 4.3: TEPcfc.py
**Source:** `mammoth_v1_backup/backbone/TEPcfc.py`
**Target:** `mammoth/backbone/TEPcfc.py`

**Required Changes:**
- Input dimension for 52 TEP variables
- Output dimension for 22 fault classes
- Sequence length handling

### Step 4.4: Update __init__.py
```python
# mammoth/backbone/__init__.py
# Add CfC backbone imports
```

## Phase 5: Port Custom Datasets

### Step 5.1: Tennessee Eastman Process
**Source:** `mammoth_v1_backup/datasets/tennessee_eastman.py`
**Target:** `mammoth/datasets/tennessee_eastman.py`

**Check:**
- ContinualDataset base class API
- get_data_loaders() signature
- Store attribute structure
- Transform handling

### Step 5.2: Update MNIST Datasets
**Files to modify:**
- `mammoth/datasets/seq_mnist.py`
- `mammoth/datasets/perm_mnist.py`
- `mammoth/datasets/rot_mnist.py`

**Changes:**
- Import MNISTcfc instead of MNISTMLP
- Verify backbone selection logic
- Update get_backbone() calls

### Step 5.3: Update __init__.py
```python
# mammoth/datasets/__init__.py
# Add tennessee-eastman dataset
```

## Phase 6: Dependencies

### Step 6.1: Update pyproject.toml or requirements
**Add to dependencies:**
```toml
ncps = "^0.1.0"  # For CfC networks
```

### Step 6.2: Install with uv
```bash
cd mammoth
uv sync  # Use v2.0's native dependency management
```

### Step 6.3: Verify ncps Installation
```bash
python -c "import ncps; print(ncps.__version__)"
python -c "from ncps.torch import CfC; print('CfC import OK')"
```

## Phase 7: Testing

### Step 7.1: Import Test
```bash
cd mammoth
python -c "from backbone.MNISTcfc import MNISTcfc; print('MNISTcfc OK')"
python -c "from backbone.cnn_cfc import CNNCfC; print('CNNCfC OK')"
python -c "from backbone.TEPcfc import TEPCfC; print('TEPCfC OK')"
python -c "from datasets.tennessee_eastman import TennesseeEastmanContinual; print('TEP OK')"
```

### Step 7.2: Quick Sanity Check
```bash
python utils/main.py --dataset seq-mnist --model sgd --lr 0.03 --n_epochs 1 --batch_size 32 --nowand 1
```

### Step 7.3: CfC Backbone Test
```bash
python utils/main.py --dataset seq-mnist --model sgd --lr 0.03 --n_epochs 1 --batch_size 32 --nowand 1
# Should use MNISTcfc automatically
```

## Phase 8: Update Benchmark Scripts

### Step 8.1: Check API Changes
**Potential changes in v2.0:**
- Command-line argument structure
- Dataset names
- Model names
- Output format

### Step 8.2: Update benchmark_*.py
**Files:**
- `benchmark_replay_methods.py`
- `benchmark_regularization_methods.py`
- `benchmark_all_methods.py`

**Changes:**
- Update method names if changed
- Update argument parsing
- Verify output parsing still works

### Step 8.3: Test Single Benchmark
```bash
python benchmark_all_methods.py --methods sgd --datasets seq-mnist --seeds 1
```

## Phase 9: Validation

### Step 9.1: Compare Method Lists
```bash
# v1.x methods
ls mammoth_v1_backup/models/*.py | wc -l

# v2.0 methods
ls mammoth/models/*.py | wc -l
```

### Step 9.2: Verify All v1.x Methods Exist in v2.0
```python
# Create compatibility check script
python check_method_compatibility.py
```

### Step 9.3: Run Comparison Experiment
```bash
# v2.0 on migration branch
python utils/main.py --dataset seq-mnist --model sgd --lr 0.03 --n_epochs 3 --seed 42

# Should match v1.x results (±1% due to randomness)
```

## Phase 10: Old Mammoth Benchmarks

### Step 10.1: Switch to old-mammoth Branch
```bash
git checkout old-mammoth
```

### Step 10.2: Run Comprehensive Benchmarks
```bash
# All methods on seq-mnist
./run_benchmarks.sh all
```

### Step 10.3: Save Results
```bash
# Results saved to results/benchmark_YYYYMMDD_HHMMSS.json
cp results/benchmark_*.json results/v1_benchmark_baseline.json
git add results/v1_benchmark_baseline.json
git commit -m "Baseline benchmarks with Mammoth v1.x + CfC"
```

## Phase 11: Merge Migration

### Step 11.1: Verify Migration Success
```bash
git checkout mammoth-v2-migration

# All tests pass
pytest tests/ -v

# Benchmark runs successfully
python benchmark_all_methods.py --methods sgd er --datasets seq-mnist --seeds 1
```

### Step 11.2: Merge to Main
```bash
git checkout main
git merge mammoth-v2-migration
```

### Step 11.3: Tag Release
```bash
git tag -a v2.0-migration -m "Migrated to Mammoth v2.0 with CfC backbones"
git push origin main --tags
```

## Phase 12: Post-Migration

### Step 12.1: Update Documentation
- Update README.md with v2.0 information
- Update MAMMOTH_VERSION.md
- Document new methods available

### Step 12.2: Run Full v2.0 Benchmarks
```bash
# Test all 70+ methods
./run_benchmarks.sh all
```

### Step 12.3: Compare v1 vs v2
```bash
python compare_versions.py \
  --v1-results results/v1_benchmark_baseline.json \
  --v2-results results/benchmark_latest.json
```

## Rollback Plan

If migration fails:
```bash
# Discard migration branch
git checkout main
git branch -D mammoth-v2-migration

# Continue using old-mammoth
git checkout old-mammoth
```

## Success Criteria

- ✅ All CfC backbones import successfully
- ✅ TEP dataset loads correctly
- ✅ seq-mnist with MNISTcfc achieves same accuracy as v1.x
- ✅ All v1.x methods available in v2.0
- ✅ Benchmark scripts run without errors
- ✅ Results format compatible with visualization scripts
- ✅ Access to 45+ new methods in v2.0

## Timeline

- **Phase 1-2:** 5 minutes (branching and backup)
- **Phase 3:** 5 minutes (clone v2.0)
- **Phase 4-5:** 30 minutes (port CfC customizations)
- **Phase 6:** 10 minutes (dependencies)
- **Phase 7-8:** 20 minutes (testing and benchmark updates)
- **Phase 9:** 15 minutes (validation)
- **Phase 10:** 2-3 hours (old-mammoth benchmarks in parallel)
- **Phase 11-12:** 15 minutes (merge and documentation)

**Total:** ~4 hours (including benchmark runtime)

## Risk Assessment

**Low Risk:**
- Branching strategy protects main
- old-mammoth preserved for rollback
- Incremental testing at each phase

**Medium Risk:**
- API changes in v2.0 (mitigated by validation phase)
- Dependency conflicts (mitigated by uv sync)

**High Risk:**
- None - comprehensive backup and testing strategy

## Next Steps

Execute phases 1-3 now to set up branches and clone v2.0.
