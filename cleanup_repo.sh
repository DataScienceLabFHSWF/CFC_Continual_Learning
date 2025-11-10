#!/bin/bash
# ============================================================================
# Repository Cleanup Script
# ============================================================================
# This script organizes the CfC Continual Learning repository by:
# - Removing old/duplicate validation scripts
# - Organizing benchmark results
# - Cleaning up temporary files
# - Organizing documentation
# ============================================================================

set -e

REPO_ROOT="/home/fneubuerger/CFC_Continual_Learning"
cd "$REPO_ROOT"

echo "============================================================================"
echo "CfC Continual Learning - Repository Cleanup"
echo "============================================================================"
echo ""

# Create organized directory structure
echo "Creating organized directory structure..."
mkdir -p scripts/validation
mkdir -p scripts/benchmarks
mkdir -p scripts/analysis
mkdir -p docs
mkdir -p results/validation
mkdir -p results/benchmarks
mkdir -p results/checkpoints

# ============================================================================
# Move validation scripts
# ============================================================================
echo ""
echo "Organizing validation scripts..."

# Keep only the extended validation script
if [ -f "run_full_validation_extended.sh" ]; then
    mv run_full_validation_extended.sh scripts/validation/
    echo "  ✓ Moved run_full_validation_extended.sh → scripts/validation/"
fi

# Remove old validation scripts
OLD_VALIDATION_SCRIPTS=(
    "run_full_validation.sh"
    "run_validation.sh"
    "run_validation_parallel.sh"
    "check_validation_results.sh"
)

for script in "${OLD_VALIDATION_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        rm "$script"
        echo "  ✗ Removed old script: $script"
    fi
done

# ============================================================================
# Move benchmark scripts
# ============================================================================
echo ""
echo "Organizing benchmark scripts..."

if [ -f "run_paper_benchmarks.sh" ]; then
    mv run_paper_benchmarks.sh scripts/benchmarks/
    echo "  ✓ Moved run_paper_benchmarks.sh → scripts/benchmarks/"
fi

if [ -f "benchmark_runner.py" ]; then
    mv benchmark_runner.py scripts/benchmarks/
    echo "  ✓ Moved benchmark_runner.py → scripts/benchmarks/"
fi

# ============================================================================
# Move analysis scripts
# ============================================================================
echo ""
echo "Organizing analysis scripts..."

ANALYSIS_SCRIPTS=(
    "analyze_paper_results.py"
    "interpretability_analysis.py"
    "visualize_results.py"
    "tep_gradient_boosting.py"
    "validate_tep_ml.py"
)

for script in "${ANALYSIS_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        mv "$script" scripts/analysis/
        echo "  ✓ Moved $script → scripts/analysis/"
    fi
done

# ============================================================================
# Organize test scripts
# ============================================================================
echo ""
echo "Organizing test scripts..."

if [ -f "test_backbones.py" ]; then
    mv test_backbones.py tests/
    echo "  ✓ Moved test_backbones.py → tests/"
fi

# ============================================================================
# Organize documentation
# ============================================================================
echo ""
echo "Organizing documentation..."

DOC_FILES=(
    "BENCHMARK_SYSTEM.md"
    "MAMMOTH_VERSION.md"
    "QUICK_REFERENCE.md"
    "README_V2_BENCHMARKS.md"
    "VALIDATION_RESULTS.md"
)

for doc in "${DOC_FILES[@]}"; do
    if [ -f "$doc" ]; then
        mv "$doc" docs/
        echo "  ✓ Moved $doc → docs/"
    fi
done

# ============================================================================
# Consolidate results directories
# ============================================================================
echo ""
echo "Organizing results directories..."

# Move validation results
if [ -d "validation_results" ] && [ "$(ls -A validation_results 2>/dev/null)" ]; then
    mv validation_results/* results/validation/ 2>/dev/null || true
    rmdir validation_results 2>/dev/null || true
    echo "  ✓ Consolidated validation_results → results/validation/"
fi

# Move benchmark results
if [ -d "benchmark_results" ] && [ "$(ls -A benchmark_results 2>/dev/null)" ]; then
    mv benchmark_results/* results/benchmarks/ 2>/dev/null || true
    rmdir benchmark_results 2>/dev/null || true
    echo "  ✓ Consolidated benchmark_results → results/benchmarks/"
fi

# Move paper results if they exist
if [ -d "paper_results" ]; then
    echo "  → paper_results/ kept in root (active benchmark results)"
fi

# ============================================================================
# Clean up temporary and cache files
# ============================================================================
echo ""
echo "Cleaning temporary files..."

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Removed __pycache__ directories"

# Remove Jupyter checkpoints
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Removed .ipynb_checkpoints directories"

# ============================================================================
# Update .gitignore
# ============================================================================
echo ""
echo "Updating .gitignore..."

cat >> .gitignore << 'EOF'

# Results and checkpoints
paper_results/
results/validation/
results/benchmarks/
results/checkpoints/
*.csv
*.log

# Wandb
wandb/
.wandb/

# Python cache
__pycache__/
*.pyc
*.pyo

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Environment
.venv/
venv/

# Secrets
.secrets.json

# Data
data/
*.npy
*.pkl
EOF

echo "  ✓ Updated .gitignore"

# ============================================================================
# Create README for new structure
# ============================================================================
echo ""
echo "Creating directory README files..."

cat > scripts/README.md << 'EOF'
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
EOF

cat > results/README.md << 'EOF'
# Results Directory

This directory stores experiment results organized by type.

## Structure

- **validation/** - Quick validation test results (1 epoch)
- **benchmarks/** - Full benchmark results (10+ epochs)
- **checkpoints/** - Model checkpoints from experiments

## Note

The `paper_results/` directory in the root contains active paper benchmark results.
EOF

echo "  ✓ Created README files"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================================"
echo "Cleanup Complete!"
echo "============================================================================"
echo ""
echo "New directory structure:"
echo ""
echo "CFC_Continual_Learning/"
echo "├── scripts/"
echo "│   ├── validation/          # Quick validation scripts"
echo "│   ├── benchmarks/          # Paper benchmark scripts"
echo "│   └── analysis/            # Analysis scripts"
echo "├── docs/                    # Documentation"
echo "├── results/"
echo "│   ├── validation/          # Validation results"
echo "│   ├── benchmarks/          # Benchmark results"
echo "│   └── checkpoints/         # Model checkpoints"
echo "├── configs/                 # Configuration files"
echo "├── tests/                   # Test scripts"
echo "├── mammoth/                 # Mammoth v2.0 framework"
echo "├── ncps/                    # Neural Circuit Policies"
echo "└── README.md                # Main documentation"
echo ""
echo "Old scripts removed:"
echo "  - run_full_validation.sh"
echo "  - run_validation.sh"
echo "  - run_validation_parallel.sh"
echo "  - check_validation_results.sh"
echo ""
echo "Kept in root:"
echo "  - CL_pipeline.ipynb         # Main notebook"
echo "  - README.md                 # Main documentation"
echo "  - paper_results/            # Active benchmark results"
echo ""
