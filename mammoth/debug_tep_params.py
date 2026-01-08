
import sys
import os
sys.path.append(os.getcwd())
from backbone.TEPcfc import BaseTEPCfC

def check_contiguous():
    model = BaseTEPCfC(input_size=52, num_classes=22, hidden_size=128, use_ncp_wiring=True)
    print("Checking parameters for contiguity...")
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            print(f"Parameter {name} is NOT contiguous. Shape: {param.shape}, Stride: {param.stride()}")
            try:
                param.view(-1)
            except RuntimeError as e:
                print(f"  view(-1) failed: {e}")
        else:
            # print(f"Parameter {name} is contiguous.")
            pass

if __name__ == "__main__":
    check_contiguous()
