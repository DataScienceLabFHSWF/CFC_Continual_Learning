# Mammoth v2.0 Migration Status

**Last Updated:** November 10, 2025  
**Branch:** `mammoth-v2-migration`

## ✅ Completed

### 1. Branch Setup
- ✅ Created `old-mammoth` branch (frozen v1.x for benchmarking)
- ✅ Created `mammoth-v2-migration` branch  
- ✅ Committed all pre-migration work

### 2. Mammoth v2.0 Installation
- ✅ Backed up mammoth_v1_backup/
- ✅ Cloned Mammoth v2.0 (commit: f82adca)
- ✅ Added ncps>=1.0.0 to pyproject.toml dependencies
- ✅ Installed with `uv pip install -e .`
- ✅ **70 methods available** (vs 25 in v1.x)

### 3. CfC Backbone Migration
- ✅ Copied all 3 CfC backbones to v2.0
- ✅ Updated imports (`register_backbone`, `MammothBackbone`)
- ✅ Added registration decorators:
  - `@register_backbone('mnistcfc')` → BaseMNISTcfc
  - `@register_backbone('cnn-cfc')` → CNNCfC  
  - `@register_backbone('tepcfc')` → BaseTEPCfC
  - `@register_backbone('teplstm')` → TEPLSTM
- ✅ **All 4 CfC backbones successfully registered**
- ✅ Verified: `python -c "from backbone import get_backbone_names"`
  ```
  Total backbones: 18
  CfC backbones: ['tepcfc', 'teplstm', 'cnn-cfc', 'mnistcfc']
  ```

## ⚠️ In Progress

### 4. Dataset Migration  
- ✅ Copied tennessee_eastman.py to mammoth/datasets/
- ⚠️ **Import errors** - v2.0 has different utils.conf API
- ⏳ Need to update:
  - Import statements (base_path_dataset → different in v2.0)
  - ContinualDataset base class compatibility
  - get_data_loaders() signature
  
### 5. Backbone Forward Method Updates
- ⏳ CfC backbones use `return_features` parameter
- ⏳ v2.0 expects `returnt` parameter ('out', 'features', 'both', 'all')
- ⏳ Need to update all three CfC backbone forward() methods

## 📋 TODO

### 6. Testing & Validation
- [ ] Fix dataset import errors
- [ ] Update forward() methods to use `returnt`
- [ ] Test: `python main.py --dataset seq-mnist --model sgd --backbone mnistcfc`
- [ ] Verify MNISTcfc integration with seq-mnist
- [ ] Run quick sanity check (1 epoch)

### 7. Benchmark Script Updates
- [ ] Check if benchmark_*.py scripts need updates for v2.0 API
- [ ] Update command-line arguments if changed
- [ ] Test single method run
- [ ] Update output parsing if format changed

### 8. Old Mammoth Benchmarks
- [ ] Switch to `old-mammoth` branch
- [ ] Run: `./run_benchmarks.sh all`
- [ ] Save baseline results
- [ ] Commit results to old-mammoth branch

### 9. Comparison & Merge
- [ ] Run same experiments on v2.0
- [ ] Compare v1 vs v2 results (should match within ±1%)
- [ ] Merge mammoth-v2-migration → main
- [ ] Tag release: v2.0-migration

## 🔧 Known Issues

### Issue 1: Dataset Import Error
**File:** `mammoth/datasets/tennessee_eastman.py`  
**Error:**
```python
ImportError: cannot import name 'base_path_dataset' from 'utils.conf'
```

**Solution:** Check v2.0's utils/conf.py structure and update imports

### Issue 2: Forward Method Signature
**Files:** All 3 CfC backbones  
**Current:** `forward(x, return_features=False)`  
**Required:** `forward(x, returnt='out')`  

**Solution:** Update all forward() methods to match v2.0 API

## 📊 Version Comparison

| Aspect | v1.x (old-mammoth) | v2.0 (migration) |
|--------|-------------------|------------------|
| **Methods** | 25 | 70 |
| **Backbones** | 13 + 3 CfC = 16 | 14 + 4 CfC = 18 |
| **Datasets** | ~15 | ~30 |
| **Dependencies** | manual requirements.txt | pyproject.toml + uv |
| **CfC Integration** | ✅ Working | ⚠️ Partial |
| **TEP Dataset** | ✅ Working | ⚠️ Needs fixes |
| **Main Entry** | utils/main.py | main.py |

## 🎯 Migration Strategy

### Short-term (Now)
1. Fix dataset imports (check v2.0 conf.py structure)
2. Update CfC forward() methods
3. Test single SGD run with MNISTcfc
4. Run old-mammoth benchmarks in parallel

### Medium-term (After migration)
1. Exploit 45 new methods in v2.0
2. Systematic comparison: v1 vs v2 performance
3. Update documentation with new methods

### Long-term
1. Contribute CfC backbones to upstream Mammoth
2. Write paper on CfC for continual learning
3. Explore v2.0-exclusive features

## 🚀 Next Actions

**Priority 1: Fix Dataset Imports**
```bash
# Check v2.0 utils/conf.py structure
cat mammoth/utils/conf.py | grep -A5 "base_path"

# Update tennessee_eastman.py imports accordingly
# Test: from datasets import get_dataset_names
```

**Priority 2: Update Forward Methods**
```python
# In all CfC backbones, change:
def forward(self, x, return_features=False):
    # ... existing code ...
    if return_features:
        return out, features
    return out

# To:
def forward(self, x, returnt='out'):
    # ... existing code ...
    if returnt == 'out':
        return out
    elif returnt == 'features':
        return features
    elif returnt == 'both':
        return (out, features)
    elif returnt == 'all':
        return (out, features)
    raise NotImplementedError("Unknown return type")
```

**Priority 3: Start Old Mammoth Benchmarks**
```bash
git checkout old-mammoth
./run_benchmarks.sh all  # Run in background (2-3 hours)
```

## 📝 Notes

- **ncps version:** Upgraded from 0.0.8 → 1.0.1 (good - more stable)
- **PyTorch:** Upgraded from 2.5.1 → 2.9.0 (verify CUDA compatibility)
- **Keep mammoth_v1_backup/** for reference
- **old-mammoth branch** is stable fallback if migration fails
- **All git history preserved** - can roll back anytime

## ✨ Benefits After Migration

1. **45+ new methods** to test CfC with
2. **Better tooling** (uv sync, checkpoint management)
3. **Active development** (upstream improvements)
4. **More datasets** for comprehensive evaluation
5. **Community support** (issues, PRs, discussions)

---

**Status:** 🟡 In Progress (60% complete)  
**Blocker:** Dataset import compatibility  
**ETA:** 1-2 hours to complete migration + testing
