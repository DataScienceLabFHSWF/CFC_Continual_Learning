# Bug Resolution Plan

## Issues Identified

### 1. Missing `tep_gradient_boosting.py`
**Status:** ✅ Fixed
- **Problem:** `benchmark_runner.py` and validation scripts were looking for `tep_gradient_boosting.py` in the wrong location.
- **Fix:** 
    - Updated path in `scripts/benchmarks/benchmark_runner.py`.
    - Updated path in `scripts/analysis/validate_tep_ml.py`.
    - Fixed data loading in `scripts/analysis/tep_gradient_boosting.py` (added correct default path `data/TEP` and transposition logic for `d00.dat`).

### 2. LwF Argument Error (`--temperature`)
**Status:** ✅ Fixed
- **Problem:** LwF model expects `--softmax_temp` but configuration files provided `--temperature`.
- **Fix:** Updated `configs/paper_experiments.yaml`, `configs/benchmark_config.yaml`, `configs/full_benchmark_parallel.yaml`, and `configs/cfc_only.yaml` to use `softmax_temp`.

### 3. EWC `TypeError` with `zeros_like`
**Status:** ❓ Cannot Reproduce / Verified Working
- **Problem:** Log showed `TypeError: zeros_like(): argument 'input' (position 1) must be Tensor, not int`.
- **Investigation:** 
    - Verified `EwcOn` implementation calls `self.net.get_params()`.
    - Verified `MammothBackbone` (and `TEPcfc`/`TEPLSTM` subclasses) implementation of `get_params` returns `torch.Tensor`.
    - Ran successful tests with `ewc_on` + `tepcfc`/`teplstm`.
- **Action:** Assumed fixed or transient. No code changes required as current implementation is correct.

## Verification
- Validated ML fixes with `python scripts/analysis/validate_tep_ml.py` (All Passed).
- Validated LwF and EWC with `benchmark_runner.py` and a test config (All Passed).

## Next Steps
- Re-run full benchmarks using `launch_paper_benchmarks.sh`.
