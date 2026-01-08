# Benchmark Status Report
**Last Updated:** 2026-01-08

## 1. Executive Summary

Benchmarks are currently **In Progress**. 
- **Completed Runs:** 125
- **Failed/Incomplete Runs:** 64
- **Active Experimets:** 4 (MNIST MLP baselines)

## 2. Active Experiments (Tmux)
The following experiments are currently running:
- `paper_mnist_mlp_sgd_s0`
- `paper_mnist_mlp_sgd_s1`
- `paper_mnist_mlp_sgd_s2`
- `paper_mnist_mlp_joint_s0`

## 3. Results Summary (Completed)

| Dataset | Backbone | Model | Mean Acc (%) | Std Dev | Samples |
|---------|----------|-------|--------------|---------|---------|
| **CIFAR** | **CfC** | derpp1000 | 75.26 | 1.11 | 3 |
| | | derpp200 | 60.85 | 1.96 | 3 |
| | | derpp500 | 68.55 | 1.86 | 3 |
| | | er1000 | 68.63 | 0.89 | 3 |
| | | er200 | 47.81 | 2.79 | 3 |
| | | er500 | 60.72 | 0.75 | 3 |
| | | erace1000 | 76.27 | 0.83 | 3 |
| | | erace200 | 64.51 | 1.08 | 3 |
| | | erace500 | 71.64 | 1.15 | 3 |
| | | joint | 84.39 | 1.11 | 3 |
| | | sgd | 19.66 | 0.04 | 3 |
| **CIFAR** | **ResNet** | derpp1000 | 75.29 | 1.61 | 3 |
| | | derpp200 | 63.15 | 1.42 | 3 |
| | | derpp500 | 70.50 | 0.60 | 3 |
| | | er1000 | 71.11 | 0.74 | 3 |
| | | er200 | 50.28 | 0.79 | 3 |
| | | er500 | 61.90 | 1.20 | 3 |
| | | erace1000 | 76.62 | 0.80 | 3 |
| | | erace200 | 66.10 | 0.67 | 3 |
| | | erace500 | 72.05 | 0.64 | 3 |
| | | ewc | 19.41 | 0.35 | 3 |
| | | joint | 84.66 | 0.59 | 3 |
| | | sgd | 19.63 | 0.03 | 3 |
| | | si | 18.96 | 0.75 | 3 |
| **MNIST** | **CfC** | agem | 37.16 | 10.80 | 3 |
| | | derpp200 | 84.24 | 1.03 | 3 |
| | | derpp500 | 91.41 | 1.09 | 3 |
| | | er200 | 83.09 | 1.68 | 3 |
| | | er500 | 90.24 | 1.03 | 3 |
| | | erace200 | 86.08 | 0.91 | 3 |
| | | joint | 98.20 | 0.17 | 3 |
| | | sgd | 19.89 | 0.06 | 3 |
| **MNIST** | **MLP** | agem | 41.10 | 2.15 | 3 |
| | | derpp200 | 84.13 | 1.30 | 3 |
| | | derpp500 | 90.85 | 0.06 | 3 |
| | | er200 | 78.38 | 0.98 | 3 |
| | | er500 | 86.34 | 0.41 | 3 |
| | | erace200 | 86.31 | 1.65 | 3 |
| | | ewc | 19.90 | 0.03 | 3 |
| | | joint | 97.24 | 0.18 | 2 |
| | | lwf | 20.97 | 0.88 | 3 |
| | | si | 20.23 | 0.49 | 3 |

## 4. Pending/Issues
- **TEP Benchmarks:** Many TEP runs are logically marked as "Failed/Incomplete" (e.g. `tep cfc derpp1000`, `tep lstm derpp500`). These were likely the ones affected by the bugs resolved today (ML path, LwF args, EWC issue). These should be restarted or are waiting in queue.
- **Missing Results:** `mnist mlp sgd` (currently running).

## 5. Next Actions
1. Allow current MNIST MLP runs to complete.
2. Verify TEP benchmarks start successfully after MNIST finishes (orchestrator handles this).
3. Monitor for "EWC" failures in TEP runs, although recent fix verification suggests they should pass.
