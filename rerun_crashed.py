import subprocess
import sys
import os

def run_command(cmd):
    print(f"Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
        print("Success!")
    except subprocess.CalledProcessError as e:
        print(f"Failed with error code {e.returncode}")

def main():
    # Common arguments for TEP EWC runs
    python_exec = "/home/fneubuerger/CFC_Continual_Learning/.venv/bin/python"
    
    tep_ewc_common = (
        f"{python_exec} mammoth/utils/main.py "
        "--dataset tennessee-eastman "
        "--model ewc_on "
        "--lr 0.001 "
        "--batch_size 64 "
        "--n_epochs 50 "
        "--enable_other_metrics 1 "
        "--num_workers 0 "
        "--savecheck task "
        "--e_lambda 1000 "
        "--gamma 1.0 "
        "--hidden_size 128 "
        "--num_classes 22 "
        "--num_features 52 "
    )

    # TEP EWC Runs
    tep_ewc_runs = [
        f"{tep_ewc_common} --backbone tepcfc --use_ncp_wiring 1 --seed 0",
        f"{tep_ewc_common} --backbone tepcfc --use_ncp_wiring 1 --seed 42",
        f"{tep_ewc_common} --backbone tepcfc --use_ncp_wiring 1 --seed 123",
        f"{tep_ewc_common} --backbone teplstm --seed 0",
        f"{tep_ewc_common} --backbone teplstm --seed 42",
        f"{tep_ewc_common} --backbone teplstm --seed 123",
    ]

    # Common arguments for TEP LwF runs
    tep_lwf_common = (
        f"{python_exec} mammoth/utils/main.py "
        "--dataset tennessee-eastman "
        "--model lwf "
        "--lr 0.001 "
        "--batch_size 64 "
        "--n_epochs 50 "
        "--enable_other_metrics 1 "
        "--num_workers 0 "
        "--savecheck task "
        "--alpha 0.5 "
        "--softmax_temp 2.0 "
        "--hidden_size 128 "
        "--num_classes 22 "
        "--num_features 52 "
    )

    # TEP LwF Runs
    tep_lwf_runs = [
        f"{tep_lwf_common} --backbone tepcfc --use_ncp_wiring 1 --seed 0",
        f"{tep_lwf_common} --backbone tepcfc --use_ncp_wiring 1 --seed 42",
        f"{tep_lwf_common} --backbone tepcfc --use_ncp_wiring 1 --seed 123",
        f"{tep_lwf_common} --backbone teplstm --seed 0",
        f"{tep_lwf_common} --backbone teplstm --seed 42",
        f"{tep_lwf_common} --backbone teplstm --seed 123",
    ]

    # Common arguments for ResNet CIFAR100 runs
    cifar_common = (
        f"{python_exec} mammoth/utils/main.py "
        "--dataset seq-cifar100 "
        "--model er "
        "--backbone resnet18_vanilla "
        "--lr 0.03 "
        "--batch_size 32 "
        "--n_epochs 50 "
        "--buffer_size 500 "
        "--num_workers 4 "
        "--num_classes 100 "
    )

    # ResNet CIFAR100 Runs
    cifar_runs = [
        f"{cifar_common} --seed 0",
    ]

    all_runs = tep_ewc_runs + tep_lwf_runs + cifar_runs

    print(f"Found {len(all_runs)} failed benchmarks to rerun.")
    
    for cmd in all_runs:
        run_command(cmd)

if __name__ == "__main__":
    main()
