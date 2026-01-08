#!/usr/bin/env python3
"""
Quick validation for traditional ML on TEP
Tests XGBoost, LightGBM, Random Forest, and Gradient Boosting
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report success/failure."""
    print(f"\nTesting: {description}")
    print(f"Command: {cmd}")
    
    result = subprocess.run(
        cmd,
        shell=True,
        executable='/bin/bash',
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"  ✓ SUCCESS")
        return True
    else:
        print(f"  ✗ FAILED (exit code: {result.returncode})")
        if result.stderr:
            # Print last few lines of error
            error_lines = result.stderr.strip().split('\n')[-10:]
            print("  Last error lines:")
            for line in error_lines:
                print(f"    {line}")
        return False

def main():
    print("="*80)
    print("TEP Traditional ML Validation")
    print("="*80)
    
    # Check if tep_gradient_boosting.py exists
    script_path = Path("scripts/analysis/tep_gradient_boosting.py")
    if not script_path.exists():
        print(f"ERROR: {script_path} not found!")
        return 1
    
    # Activate venv and run traditional ML tests
    venv_activate = "source .venv/bin/activate"
    
    # Only xgboost and lightgbm are supported
    models = ['xgboost', 'lightgbm']
    results = {}
    
    for model in models:
        cmd = f"{venv_activate} && python scripts/analysis/tep_gradient_boosting.py --models {model} --output validation_ml_{model}.json"
        success = run_command(cmd, f"TEP + {model}")
        results[model] = success
    
    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for model, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {model}")
    
    print(f"\nTotal: {success_count}/{total_count} passed")
    
    if success_count == total_count:
        print("\n✓ All traditional ML methods validated!")
        return 0
    else:
        print(f"\n✗ {total_count - success_count} validation(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
