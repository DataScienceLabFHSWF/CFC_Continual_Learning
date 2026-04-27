# Research Plan: Neural Circuit Policies for Continual Learning

**Date:** April 9, 2026  
**Author:** Felix Neubürger  
**Status:** Phase 2 — Ablations & Paper Finalization

---

## Quick Reference

```bash
make help              # Show all available targets
make status            # Check WandB experiment completion  
make validate          # Quick 1-epoch smoke test of all backbones
make run-tep           # Rerun TEP experiments (dataset fix applied)
make run-ablations     # Run H1/H2 ablation experiments
make run-all-missing   # Run everything that's missing
make analyze           # Pull WandB data, generate tables
make paper             # Compile LaTeX paper
make clean-repo        # Archive stale files
```

---

## Execution Order

### Step 1: Validate backbones (5 min)
```bash
make validate
```
Runs 1-epoch SGD on all 9 backbones (CfC/LTC/RandomSparse × MNIST/CIFAR/TEP) to confirm they load and train without errors.

### Step 2: Launch TEP re-runs (parallel, ~60h with 4 GPUs)
```bash
make run-tep
```
All 42 previous TEP runs used the broken dataset loader (`self.windows` instead of `self.data`). The fix (commit 311cb86) is applied. This deletes old TEP logs and forces a clean re-run.

**Note:** TEP uses `N_CLASSES_PER_TASK=1` (22 tasks, 1 fault each), so Task-IL will be trivially 100%. Report only Class-IL for TEP.

### Step 3: Launch ablation experiments (parallel, ~3-5 days)
```bash
make run-ablations       # Both MNIST and CIFAR ablations
```
This runs:
- **MNIST**: `mnistltc` + `mnist-random-sparse` × 6 methods × 3 seeds = **36 runs**
- **CIFAR**: `cnn-ltc` + `cnn-random-sparse` × 7 methods × 3 seeds = **42 runs**
- **Total: 78 new runs**

Methods per backbone:
| | SGD | Joint | ER-200 | ER-500 | ER-1000 | DER++-500 | ER-ACE |
|---|---|---|---|---|---|---|---|
| MNIST | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓(200) |
| CIFAR | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓(500) |

### Step 4: Monitor progress
```bash
make status              # WandB overview
tmux ls                  # Active tmux sessions
tail -f paper_results/logs/<experiment>.log  # Live log
```

### Step 5: Analyze results
```bash
make analyze             # Pull all WandB data to CSV
```
Then update paper tables manually or via `scripts/analysis/analyze_wandb_results.py`.

### Step 6: Compile paper
```bash
make paper               # pdflatex + bibtex + 2× pdflatex
```

---

## 1. Project Status Assessment

### What We Have (Completed)
- **247 finished WandB runs** across MNIST, CIFAR-10, and TEP
- **12 backbone implementations** (CfC, LTC, RandomSparse, Dense × MNIST/CIFAR/TEP + LSTM, MLP)
- **Paper draft** (10-page LaTeX) — fully rewritten with honest claims and verified data
- **Full infrastructure**: Makefile, tmux orchestrator, WandB logging, skip detection
- **All 3 LaTeX tables** fixed with clean WandB data

### What Works
| Benchmark | Finding | Confidence |
|-----------|---------|------------|
| MNIST (CfC vs MLP) | CfC outperforms MLP by +3-5% in low-buffer ER | ✅ High (3 seeds) |
| MNIST (DER++/ER-ACE) | CfC ≈ MLP when method is strong | ✅ High |
| CIFAR-10 (CfC vs ResNet) | ResNet slightly better in ER; ties at large buffers | ✅ High |
| CIFAR-10 (Joint) | CfC matches ResNet upper bound (84.4 vs 84.7) | ✅ High |

### What Was Broken → Fixed
| Issue | Status | Resolution |
|-------|--------|------------|
| TEP dataset loader | ✅ Fixed | Used `self.data`/`self.targets` instead of `self.windows`/`self.labels` (commit 311cb86) |
| TEP ablation backbones crash | ✅ Fixed | Reset `hidden_state=None` each forward pass (batch size mismatch) |
| LaTeX tables rendering | ✅ Fixed | Rewritten from clean WandB data |
| Paper overclaiming | ✅ Fixed | Rewrote abstract, contributions, experiments, conclusion |
| Skip detection broken | ✅ Fixed | Check for `wandb: Synced` marker (not just completion echo) |
| xcolor/rowcolor LaTeX errors | ✅ Fixed | Added `dvipsnames,table` to xcolor package |

### Still Pending
| Issue | Severity | Details |
|-------|----------|---------|
| **TEP results need re-running** | 🔴 HIGH | All 42 TEP runs used broken dataset loader |
| **LTC/RandomSparse ablations** | 🟡 HIGH | Never ran — H1 and H2 hypotheses untested |
| **Advanced metrics** | 🟡 MEDIUM | Tau monitoring, gradient analysis not collected |

---

## 2. Research Questions

### Primary Research Question
**RQ: Can brain-inspired architectural inductive biases (sparse wiring + continuous-time dynamics) inherently reduce catastrophic forgetting in neural networks?**

### Sub-Questions (mapped to hypotheses)

**RQ1 (Modularity — H1):** Does the structured sparse connectivity of AutoNCP wiring reduce task interference compared to (a) random sparse and (b) fully connected architectures?
- *Metric:* Class-IL accuracy, Backward Transfer (BWT)
- *Status:* ❌ **Untested** — LTC and RandomSparse backbones implemented but never benchmarked

**RQ2 (Temporal Stability — H2):** Do the adaptive time constants (τ) of CfC/LTC neurons create a functional separation between stable "memory" neurons and plastic "adaptation" neurons?
- *Metric:* τ distribution bimodality coefficient, representational stability
- *Status:* ❌ **Untested** — tau_monitor.py exists but was never integrated

**RQ3 (Gradient Isolation — H3):** Does sparse NCP wiring lead to more orthogonal gradients between tasks compared to dense networks?
- *Metric:* Cosine similarity of inter-task gradients
- *Status:* ❌ **Untested** — advanced_metrics.py exists but was never integrated

**RQ4 (Practical Value):** Does CfC provide practical advantages over standard backbones (MLP, ResNet) when combined with existing CL methods?
- *Metric:* Class-IL and Task-IL accuracy across methods and buffer sizes
- *Status:* ✅ **Answered** — MNIST yes (+3-5%), CIFAR-10 competitive but not better

### Additional Questions (from data analysis)

**RQ5:** Why does CfC show larger advantages in low-buffer regimes (ER-200) than high-buffer regimes (ER-1000)?
- *Hypothesis:* Network stability compensates for less replay data
- *Status:* 🟡 Observed but not explained mechanistically

**RQ6:** Why does TEP fail completely? Is this a data loader bug, evaluation bug, or fundamental architectural mismatch?
- *Status:* 🔴 Must be diagnosed before any TEP claims

---

## 3. Honest Assessment: What Can We Claim?

### Strong Claims (supported by data)
1. **CfC architectures are viable backbones for continual learning** — they achieve competitive performance with standard architectures on two benchmarks
2. **CfC shows particular advantage in low-buffer replay** — consistent +3-5% on MNIST with ER-200/500
3. **The CfC advantage diminishes with stronger methods** — DER++ and ER-ACE equalize performance, suggesting CfC's implicit regularization overlaps with explicit replay
4. **On CIFAR-10, replacing the FC head with a CfC head does not help** — ResNet is marginally better in most configs

### Claims We Cannot Make (yet)
1. ❌ "NCPs reduce forgetting through modularity" — never tested RandomSparse vs AutoNCP
2. ❌ "Time constants create stability-plasticity separation" — tau data never collected
3. ❌ "Gradients are more orthogonal in NCPs" — gradient analysis never ran
4. ❌ "CfC outperforms on temporal tasks" — TEP results are broken
5. ❌ "Architectural bias is an alternative to replay" — SGD baseline is random chance for both

### The Narrative We Can Write
**"We evaluated NCP/CfC architectures on continual learning benchmarks and found they provide modest but consistent advantages in low-resource replay settings. The CfC backbone acts as an implicit regularizer, partially substituting for explicit replay data. However, this advantage diminishes when combined with stronger CL methods, suggesting the mechanisms overlap rather than complement."**

This is an honest, publishable finding — it establishes a baseline and opens the door for mechanistic investigation.

---

## 4. Detailed Plan: Path to a Publishable Paper

### Phase 1: Fix Critical Issues (1-2 days)

#### 1.1 Diagnose and Fix TEP
- [ ] Check TEP dataset loader (`tennessee_eastman.py`) for evaluation mask bugs
- [ ] Verify TEP data preprocessing (normalization, class labels, task boundaries)
- [ ] Run a single TEP-CfC experiment with verbose logging
- [ ] If fixable: rerun TEP suite (48 experiments, ~2 days)
- [ ] If unfixable: **drop TEP from the paper** and focus on MNIST + CIFAR-10

#### 1.2 Fix LaTeX Tables
- [ ] Regenerate tables from raw WandB data (not from buggy pandas export)
- [ ] Add Task-IL columns alongside Class-IL
- [ ] Add BWT (Backward Transfer) column

### Phase 2: Run Missing Ablations (3-5 days)

#### 2.1 Wiring Ablation (H1 — Modularity)
Run on MNIST and CIFAR-10 only:
```
Backbones: mnist_random_sparse, mnist_dense_cfc, cnn_random_sparse, cnn_dense_cfc
Methods: SGD, ER-200, ER-500, DER++-500, Joint
Seeds: 0, 1, 2
Total: 2 datasets × 2 backbones × 5 methods × 3 seeds = 60 runs
```

#### 2.2 LTC vs CfC Ablation (H2 — Dynamics)
```
Backbones: mnistltc, cnn-ltc
Methods: SGD, ER-200, ER-500, DER++-500, Joint
Seeds: 0, 1, 2
Total: 2 datasets × 1 backbone × 5 methods × 3 seeds = 30 runs
```
Note: LTC is 10-20× slower than CfC due to ODE solver

#### 2.3 Advanced Metrics Collection (H2, H3)
- [ ] Integrate tau_monitor.py into training loop
- [ ] Integrate gradient_interference from advanced_metrics.py
- [ ] Run representative subset (ER-500 on MNIST: MLP, CfC, RandomSparse, Dense) with metrics
- [ ] 4 backbones × 1 method × 3 seeds = 12 runs with metrics logging

### Phase 3: Analysis & Visualization (2-3 days)

- [ ] Download all WandB data to local CSV
- [ ] Generate publication-quality tables (Class-IL, Task-IL, BWT, ±std)
- [ ] Create comparison plots:
  - Bar chart: Method × Backbone for each dataset
  - Line plot: Buffer size vs accuracy for CfC and baseline
  - Spider/radar plot: Multi-metric comparison (optional)
- [ ] If H1 tested: Wiring comparison bar chart
- [ ] If H2 tested: τ distribution histograms, bimodality analysis
- [ ] Statistical significance tests (paired t-test or Wilcoxon)

### Phase 4: Paper Writing (3-5 days)

#### Paper Structure (target: 8-10 pages, NeurIPS/ICML format)
1. **Abstract** — Rewrite with actual results
2. **Introduction** — Frame as "first evaluation of NCPs for CL"
3. **Related Work** — Current draft is good, minor updates
4. **Background** — Current draft is good
5. **Method** — Current draft is good, add training details
6. **Experiments** — REWRITE with real tables and analysis
7. **Discussion** — NEW: honest analysis of when CfC helps and when it doesn't
8. **Conclusion** — Grounded claims + future work

### Phase 5: Polish (1-2 days)
- [ ] Proofread
- [ ] Verify all numbers match WandB
- [ ] Check references
- [ ] Generate final figures
- [ ] Write reproducibility appendix

---

## 5. Decision Points

### Decision 1: TEP — Include or Drop?
- **If fixable in 1 day:** Include. Temporal data is the strongest case for CfC.
- **If not fixable:** Drop and mention as future work. Paper is still strong with MNIST + CIFAR.

### Decision 2: Ablations — Run or Analyze What We Have?
- **Option A (Full ablations):** +1-2 weeks but much stronger paper (tests H1-H3)
- **Option B (Current data only):** Paper is limited to "CfC vs baseline" comparison
- **Recommendation:** Run at least the RandomSparse ablation (H1) — it's the most novel contribution

### Decision 3: Venue
- **Workshop paper (4 pages):** Can submit now with MNIST + CIFAR results only
- **Full paper (8-10 pages):** Needs ablations and at least one mechanistic hypothesis tested
- **Journal:** Needs all hypotheses + TEP + deep analysis

---

## 6. Minimum Viable Paper (can write NOW)

Even without ablations, we can write a valid paper with:
1. ✅ MNIST results (MLP vs CfC × 7 methods × 3 seeds)
2. ✅ CIFAR results (ResNet vs CfC × 10 methods × 3 seeds)
3. ✅ Analysis of when CfC helps (low-buffer replay)
4. ✅ Analysis of when it doesn't (strong methods equalize)
5. ✅ Parameter count comparison (CfC is ~60% smaller)
6. ❌ No mechanistic explanation (H1-H3 untested)

**Title option:** "Evaluating Neural Circuit Policies as Backbones for Continual Learning"

This positions it as a systematic empirical study rather than a mechanistic paper.

---

## 7. Timeline

| Week | Task | Deliverable |
|------|------|------------|
| Week 1 (Apr 9-15) | Fix TEP, fix tables, start paper rewrite | Working TEP or decision to drop |
| Week 2 (Apr 16-22) | Run H1 ablation (RandomSparse), analysis | Ablation results in WandB |
| Week 3 (Apr 23-29) | Complete analysis, finish paper draft | Full draft with tables + figures |
| Week 4 (Apr 30-May 6) | Polish, review, submit | Camera-ready paper |
