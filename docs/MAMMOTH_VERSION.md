# Mammoth Version Information

## Current Version (In Use)

**Source:** Custom fork based on Mammoth v1.x  
**Location:** `/home/fneubuerger/CFC_Continual_Learning/mammoth/`  
**Status:** ✅ Active - Modified for CfC integration

### Available Methods (25 total):
- **Baselines:** SGD, Joint, Joint-GCL
- **Replay:** ER, DER, DER++, GDumb, GSS, HAL, iCaRL, MER, ER-ACE, FDR
- **X-DER variants:** X-DER, X-DER-CE, X-DER-RPC
- **Regularization:** EWC-Online, SI, LwF, LwF-MC
- **Architecture:** PNN, RPC
- **Distillation:** BiC, LUCiR
- **Meta:** A-GEM, A-GEM-R, GEM

### Custom Modifications:
✅ **CfC Backbones Added:**
- `backbone/MNISTcfc.py` - AutoNCP for sequential MNIST (23K params)
- `backbone/cnn_cfc.py` - ResNet18 + CfC (1.33M params)
- `backbone/TEPcfc.py` - Industrial process monitoring

✅ **Datasets Extended:**
- `datasets/seq_mnist.py` - Updated to use MNISTcfc
- `datasets/perm_mnist.py` - Updated to use MNISTcfc
- `datasets/rot_mnist.py` - Updated to use MNISTcfc
- `datasets/tennessee_eastman.py` - New TEP dataset (22 faults)

✅ **Testing Infrastructure:**
- Unit tests for all CfC backbones
- TEP incremental vs joint experiments
- Sanity checks for Mammoth integration

---

## Mammoth v2.0 (Available for Migration)

**Source:** https://github.com/aimagelab/mammoth  
**Status:** ⏳ Available but not yet integrated

### New Features:
- **70+ methods** (vs current 25)
- **Better tooling:** `uv sync` for dependencies
- **Checkpoint management:** Automatic model saving/loading
- **SIGINT handling:** Graceful interruption
- **Improved logging:** Better WandB integration
- **More datasets:** Additional benchmarks

### Additional Methods in v2.0:
- More replay variants
- Advanced meta-learning approaches
- Newer regularization techniques
- Task-free continual learning methods
- Generative replay methods

---

## Migration Considerations

### Option 1: Stay with Current Version ✅ (Recommended for now)
**Pros:**
- Already customized with CfC backbones
- All benchmarking scripts ready
- TEP dataset integrated
- Tested and working

**Cons:**
- Missing 45+ newer methods
- Less mature tooling
- Manual dependency management

### Option 2: Parallel Installation
Install Mammoth v2.0 alongside current version:
```bash
# Clone new Mammoth
git clone https://github.com/aimagelab/mammoth.git mammoth_v2
cd mammoth_v2
uv sync

# Keep current mammoth/ with CfC modifications
# Use mammoth_v2/ for additional experiments
```

**Pros:**
- Access to all 70+ methods
- No risk to current working setup
- Can compare versions

**Cons:**
- Duplicate code/experiments
- Need to port CfC backbones to v2.0
- Maintenance overhead

### Option 3: Full Migration (Future)
Replace current mammoth/ with v2.0 and re-integrate CfC:
```bash
# Backup current work
mv mammoth mammoth_v1_backup

# Install v2.0
git clone https://github.com/aimagelab/mammoth.git
cd mammoth

# Port CfC modifications
cp ../mammoth_v1_backup/backbone/*cfc.py backbone/
cp ../mammoth_v1_backup/datasets/tennessee_eastman.py datasets/
# Update imports and dependencies
```

**Pros:**
- Access to latest methods
- Better infrastructure
- Future-proof

**Cons:**
- Significant integration work
- Risk of breaking changes
- Need to test all CfC modifications

---

## Recommendation

### Short-term (Current Phase):
**✅ Use current Mammoth v1.x with CfC modifications**

1. Complete benchmarking with existing 25 methods
2. Implement Bayesian CfC methods
3. Run TEP experiments
4. Publish initial results

### Medium-term (After Initial Results):
**🔄 Parallel installation for comparison**

1. Install Mammoth v2.0 in separate directory
2. Port CfC backbones to v2.0
3. Run comparative benchmarks
4. Identify v2.0-exclusive methods of interest

### Long-term (Next Research Phase):
**⬆️ Full migration to Mammoth v2.0**

1. Complete migration once v2.0 stabilizes
2. Contribute CfC backbones back to Mammoth repository
3. Leverage all 70+ methods for comprehensive study
4. Benefit from community improvements

---

## Current Status

**Active Version:** Mammoth v1.x (customized)  
**Next Action:** Complete benchmarking with current 25 methods  
**Future Upgrade:** Migrate to v2.0 after initial experiments complete

The benchmarking scripts (`benchmark_*.py`) are configured for the **current version** and will work out of the box.
