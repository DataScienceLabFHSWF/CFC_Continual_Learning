#!/bin/bash
set -e

# Activate virtual environment
source /home/fneubuerger/CFC_Continual_Learning/.venv/bin/activate

# Set WandB environment variables
export WANDB_ENTITY="fneubuerger"
export WANDB_PROJECT="mammoth"

echo "Starting rerun of crashed benchmarks..."

# --- TEP EWC Runs ---
echo "Running TEP EWC (TEP-CfC)..."
python mammoth/utils/main.py --dataset tennessee-eastman --model ewc_on --lr 0.001 --batch_size 64 --n_epochs 50 --enable_other_metrics 1 --num_workers 0 --savecheck task --e_lambda 1000 --gamma 1.0 --hidden_size 128 --num_classes 22 --num_features 52 --backbone tepcfc --use_ncp_wiring 1 --seed 0
python mammoth/utils/main.py --dataset tennessee-eastman --model ewc_on --lr 0.001 --batch_size 64 --n_epochs 50 --enable_other_metrics 1 --num_workers 0 --savecheck task --e_lambda 1000 --gamma 1.0 --hidden_size 128 --num_classes 22 --num_features 52 --backbone tepcfc --use_ncp_wiring 1 --seed 42
python mammoth/utils/main.py --dataset tennessee-eastman --model ewc_on --lr 0.001 --batch_size 64 --n_epochs 50 --enable_other_metrics 1 --num_workers 0 --savecheck task --e_lambda 1000 --gamma 1.0 --hidden_size 128 --num_classes 22 --num_features 52 --backbone tepcfc --use_ncp_wiring 1 --seed 123

echo "Running TEP EWC (TEP-LSTM)..."
python mammoth/utils/main.py --dataset tennessee-eastman --model ewc_on --lr 0.001 --batch_size 64 --n_epochs 50 --enable_other_metrics 1 --num_workers 0 --savecheck task --e_lambda 1000 --gamma 1.0 --hidden_size 128 --num_classes 22 --num_features 52 --backbone teplstm --seed 0
python mammoth/utils/main.py --dataset tennessee-eastman --model ewc_on --lr 0.001 --batch_size 64 --n_epochs 50 --enable_other_metrics 1 --num_workers 0 --savecheck task --e_lambda 1000 --gamma 1.0 --hidden_size 128 --num_classes 22 --num_features 52 --backbone teplstm --seed 42
python mammoth/utils/main.py --dataset tennessee-eastman --model ewc_on --lr 0.001 --batch_size 64 --n_epochs 50 --enable_other_metrics 1 --num_workers 0 --savecheck task --e_lambda 1000 --gamma 1.0 --hidden_size 128 --num_classes 22 --num_features 52 --backbone teplstm --seed 123

# --- TEP ER Runs ---
echo "Running TEP ER (TEP-CfC)..."
python mammoth/utils/main.py --dataset tennessee-eastman --model er --backbone tepcfc --n_epochs 50 --batch_size 64 --num_workers 0 --savecheck task --enable_other_metrics 1 --lr 0.001 --buffer_size 1000 --minibatch_size 64 --num_features 52 --num_classes 22 --hidden_size 128 --use_ncp_wiring 1 --seed 0
python mammoth/utils/main.py --dataset tennessee-eastman --model er --backbone tepcfc --n_epochs 50 --batch_size 64 --num_workers 0 --savecheck task --enable_other_metrics 1 --lr 0.001 --buffer_size 1000 --minibatch_size 64 --num_features 52 --num_classes 22 --hidden_size 128 --use_ncp_wiring 1 --seed 42
python mammoth/utils/main.py --dataset tennessee-eastman --model er --backbone tepcfc --n_epochs 50 --batch_size 64 --num_workers 0 --savecheck task --enable_other_metrics 1 --lr 0.001 --buffer_size 1000 --minibatch_size 64 --num_features 52 --num_classes 22 --hidden_size 128 --use_ncp_wiring 1 --seed 123

echo "Running TEP ER (TEP-LSTM)..."
python mammoth/utils/main.py --dataset tennessee-eastman --model er --backbone teplstm --n_epochs 50 --batch_size 64 --num_workers 0 --savecheck task --enable_other_metrics 1 --lr 0.001 --buffer_size 1000 --minibatch_size 64 --num_features 52 --num_classes 22 --hidden_size 128 --seed 0
python mammoth/utils/main.py --dataset tennessee-eastman --model er --backbone teplstm --n_epochs 50 --batch_size 64 --num_workers 0 --savecheck task --enable_other_metrics 1 --lr 0.001 --buffer_size 1000 --minibatch_size 64 --num_features 52 --num_classes 22 --hidden_size 128 --seed 42
python mammoth/utils/main.py --dataset tennessee-eastman --model er --backbone teplstm --n_epochs 50 --batch_size 64 --num_workers 0 --savecheck task --enable_other_metrics 1 --lr 0.001 --buffer_size 1000 --minibatch_size 64 --num_features 52 --num_classes 22 --hidden_size 128 --seed 123

# --- Image ER Runs ---
echo "Running ER (CIFAR100)..."
python mammoth/utils/main.py --dataset seq-cifar100 --model er --backbone resnet18_vanilla --lr 0.03 --batch_size 32 --n_epochs 50 --buffer_size 500 --num_workers 4 --seed 0

echo "Running ER (CIFAR10)..."
python mammoth/utils/main.py --dataset seq-cifar10 --model er --backbone resnet18_vanilla --num_workers 4 --n_epochs 50 --batch_size 32 --buffer_size 500 --lr 0.03 --seed 0

echo "Running ER (MNIST)..."
python mammoth/utils/main.py --dataset seq-mnist --model er --backbone mnistmlp --num_workers 4 --n_epochs 10 --batch_size 32 --buffer_size 200 --lr 0.03 --seed 0

# --- HOPE Runs ---
echo "Running HOPE (CIFAR100)..."
python mammoth/utils/main.py --dataset seq-cifar100 --model hope --backbone hope --lr 0.001 --hope_lr 0.001 --batch_size 32 --n_epochs 50 --input_size 32 --input_channels 3 --seed 0

echo "Running HOPE (CIFAR10)..."
python mammoth/utils/main.py --dataset seq-cifar10 --model hope --backbone hope --lr 0.001 --hope_lr 0.001 --batch_size 32 --n_epochs 50 --input_size 32 --input_channels 3 --seed 0

echo "Running HOPE (MNIST)..."
python mammoth/utils/main.py --dataset seq-mnist --model hope --backbone hope --lr 0.001 --hope_lr 0.001 --batch_size 32 --n_epochs 10 --input_size 28 --input_channels 1 --seed 0

echo "All reruns completed successfully!"
