#!/bin/bash
# Setup script for CFC Continual Learning experiments

# Load secrets if they exist
if [ -f ".secrets.json" ]; then
    export WANDB_API_KEY=$(python3 -c "import json; print(json.load(open('.secrets.json'))['wandb_api_key'])")
    export WANDB_ENTITY=$(python3 -c "import json; print(json.load(open('.secrets.json'))['wandb_entity'])")
    export WANDB_PROJECT=$(python3 -c "import json; print(json.load(open('.secrets.json'))['wandb_project'])")
    echo "✓ Loaded secrets from .secrets.json"
else
    echo "⚠ No .secrets.json found. Copy .secrets.json.template and fill in your credentials."
    echo "  cp .secrets.json.template .secrets.json"
    exit 1
fi

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "✓ Activated virtual environment"
    else
        echo "⚠ No virtual environment found. Create one with: uv venv .venv"
        exit 1
    fi
fi

# Check if dependencies are installed
if ! python -c "import torch" 2>/dev/null; then
    echo "⚠ Dependencies not installed. Installing..."
    uv pip install -r requirements.txt
fi

echo "✓ Environment ready"
echo ""
echo "Usage examples:"
echo "  cd mammoth"
echo "  python utils/main.py --dataset seq-mnist --model sgd --lr 0.03 --n_epochs 5 --batch_size 32"
echo "  python utils/main.py --dataset seq-mnist --model ewc_on --lr 0.03 --n_epochs 5 --batch_size 32 --e_lambda 0.1 --gamma 1.0"
