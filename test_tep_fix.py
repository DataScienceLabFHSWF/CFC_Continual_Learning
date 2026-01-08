
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from mammoth.backbone.TEPcfc import BaseTEPCfC

def test_get_params():
    print("Initializing model...")
    try:
        model = BaseTEPCfC(input_size=52, num_classes=22, hidden_size=128, use_ncp_wiring=True)
    except ImportError:
        print("NCPS not installed, skipping test")
        return

    print("Model initialized.")
    
    # Move to GPU if available (to match training conditions)
    if torch.cuda.is_available():
        model = model.cuda()
        print("Moved to CUDA.")

    print("Calling get_params()...")
    try:
        params = model.get_params()
        print(f"get_params() successful. Shape: {params.shape}")
    except Exception as e:
        print(f"get_params() FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_get_params()
