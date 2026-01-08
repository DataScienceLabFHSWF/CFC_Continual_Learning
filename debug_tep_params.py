
import sys
import os
import torch

# Add current directory to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'mammoth'))

from mammoth.backbone.TEPcfc import BaseTEPCfC

def test_get_params():
    print("Instantiating BaseTEPCfC...")
    try:
        model = BaseTEPCfC(input_size=52, num_classes=22, hidden_size=128)
        print("Model instantiated.")
    except ImportError as e:
        print(f"ImportError: {e}")
        return

    params = model.get_params()
    print(f"get_params returned type: {type(params)}")
    if isinstance(params, torch.Tensor):
        print(f"get_params shape: {params.shape}")
    else:
        print(f"get_params value: {params}")

    try:
        z = torch.zeros_like(params)
        print("torch.zeros_like(params) successful")
    except Exception as e:
        print(f"torch.zeros_like(params) failed: {e}")

if __name__ == "__main__":
    test_get_params()
