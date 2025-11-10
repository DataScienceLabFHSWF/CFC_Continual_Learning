# Scripts Directory

This directory contains all executable scripts for the CfC Continual Learning project.

## Structure

- **validation/** - Quick validation scripts (1 epoch tests)
- **benchmarks/** - Full paper benchmark scripts (10+ epochs)
- **analysis/** - Result analysis and visualization scripts

## Usage

### Validation
```bash
./validation/run_full_validation_extended.sh
```

### Paper Benchmarks
```bash
./benchmarks/run_paper_benchmarks.sh [--dataset all|mnist|cifar10|tep]
```

### Analysis
```bash
python analysis/analyze_paper_results.py --results-dir ../paper_results
```
