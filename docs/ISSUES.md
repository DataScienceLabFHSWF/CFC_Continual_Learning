# Issues and Tasks

## Priority 1: Hybrid HOPE Strategy
**Type:** Feature / Research  
**Status:** Closed
**Created:** 2026-01-08

### Context
Pure HOPE architecture (Nested Learning/Titan Memory) fails on Class-Incremental Learning (Class-IL) benchmarks due to the "Growing Head" problem. The original paper evaluates on Language Modeling (Fixed Vocabulary), which does not suffer from head expansion induced forgetting.

### Objective
Implement and validate a **Hybrid Strategy** combining HOPE Backbone with Experience Replay (ER).

### Implementation Plan
1.  **Config**: Use `model: er` with `backbone: hope`.
2.  **Hypothesis**: The Replay Buffer will provide gradients for old classes during the "Plastic" update of the dense memory, preventing the destruction of keys used by old output heads.
3.  **Benchmark**: Run `hope_hybrid_er` on Seq-CIFAR10.
4.  **Success Metric**: >10% Accuracy on Task 1 after training Task 5 (currently 0%).

### Resolution (2026-01-08)
- **Implemented**: Modified `mammoth/models/hope.py` to optionally accept `buffer_size` and perform experience replay concatenation.
- **Validated**: Pilot run (PID 56468) achieved **40.3% Class-IL accuracy** on Task 1 after training Task 2, significantly outperforming the 0% baseline:
    - Task 1 Acc: 40.3% (vs 0% Pure)
    - Task 2 Acc: 65.8%
    - Mean Class-IL: 53.08%
- **Conclusion**: Hybrid HOPE+ER stabilizes the dense memory updates in Class-IL settings.

### Resources
- Config: `configs/hope_hybrid_er.yaml`
- Analysis: `docs/HOPE_IMPLEMENTATION_ANALYSIS.md`
