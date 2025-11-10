#!/bin/bash
# Extended CfC Validation - CfC backbones + Standard baselines for comparison

set -e

cd /home/fneubuerger/CFC_Continual_Learning

# Create results directory
mkdir -p validation_results

echo "=========================================================="
echo "CFC Continual Learning - Extended Validation Suite"
echo "=========================================================="
echo ""
echo "Starting validation at $(date)"
echo ""
echo "CfC Backbone Tests:"
echo "  1. seq-mnist + mnistcfc (ER)"
echo "  2. seq-cifar10 + cnn-cfc (ER)"
echo "  3. tennessee-eastman + tepcfc (ER)"
echo "  4. tennessee-eastman + teplstm (ER)"
echo ""
echo "Standard Baselines for Comparison:"
echo "  5. seq-mnist + mnistmlp (SGD)"
echo "  6. seq-mnist + mnistmlp (ER)"
echo "  7. seq-mnist + mnistmlp (DER++)"
echo "  8. seq-mnist + mnistmlp (ER-ACE)"
echo ""
echo "Each test runs in a separate tmux session"
echo "Use 'tmux ls' to see active sessions"
echo "Use 'tmux attach -t <session>' to view progress"
echo ""

# Kill any existing validation sessions
tmux kill-session -t val_mnist 2>/dev/null || true
tmux kill-session -t val_cifar 2>/dev/null || true
tmux kill-session -t val_tep_cfc 2>/dev/null || true
tmux kill-session -t val_tep_lstm 2>/dev/null || true
tmux kill-session -t val_baseline_sgd 2>/dev/null || true
tmux kill-session -t val_baseline_er 2>/dev/null || true
tmux kill-session -t val_baseline_derpp 2>/dev/null || true
tmux kill-session -t val_baseline_ace 2>/dev/null || true

sleep 1

echo "========================================"
echo "Starting CfC Backbone Tests..."
echo "========================================"

# 1. MNIST + mnistcfc
echo "[1/8] Starting MNIST + mnistcfc (CfC) validation..."
tmux new-session -d -s val_mnist "cd /home/fneubuerger/CFC_Continual_Learning/mammoth && \
source ../.venv/bin/activate && \
python utils/main.py \
  --dataset seq-mnist \
  --model er \
  --backbone mnistcfc \
  --n_epochs 1 \
  --batch_size 32 \
  --lr 0.03 \
  --buffer_size 200 \
  --num_workers 0 \
  --seed 0 \
  2>&1 | tee ../validation_results/mnist_mnistcfc.log; \
echo 'MNIST CfC validation completed' && \
read -p 'Press enter to close...'"

# 2. CIFAR-10 + cnn-cfc
echo "[2/8] Starting CIFAR-10 + cnn-cfc (CfC) validation..."
tmux new-session -d -s val_cifar "cd /home/fneubuerger/CFC_Continual_Learning/mammoth && \
source ../.venv/bin/activate && \
python utils/main.py \
  --dataset seq-cifar10 \
  --model er \
  --backbone cnn-cfc \
  --n_epochs 1 \
  --batch_size 32 \
  --lr 0.03 \
  --buffer_size 200 \
  --num_workers 0 \
  --seed 0 \
  2>&1 | tee ../validation_results/cifar_cnncfc.log; \
echo 'CIFAR CfC validation completed' && \
read -p 'Press enter to close...'"

# 3. TEP + tepcfc
echo "[3/8] Starting TEP + tepcfc (CfC) validation..."
tmux new-session -d -s val_tep_cfc "cd /home/fneubuerger/CFC_Continual_Learning/mammoth && \
source ../.venv/bin/activate && \
python utils/main.py \
  --dataset tennessee-eastman \
  --model er \
  --backbone tepcfc \
  --n_epochs 1 \
  --batch_size 32 \
  --lr 0.001 \
  --buffer_size 200 \
  --num_workers 0 \
  --seed 0 \
  2>&1 | tee ../validation_results/tep_tepcfc.log; \
echo 'TEP-CfC validation completed' && \
read -p 'Press enter to close...'"

# 4. TEP + teplstm
echo "[4/8] Starting TEP + teplstm validation..."
tmux new-session -d -s val_tep_lstm "cd /home/fneubuerger/CFC_Continual_Learning/mammoth && \
source ../.venv/bin/activate && \
python utils/main.py \
  --dataset tennessee-eastman \
  --model er \
  --backbone teplstm \
  --n_epochs 1 \
  --batch_size 32 \
  --lr 0.001 \
  --buffer_size 200 \
  --num_workers 0 \
  --seed 0 \
  2>&1 | tee ../validation_results/tep_teplstm.log; \
echo 'TEP-LSTM validation completed' && \
read -p 'Press enter to close...'"

echo ""
echo "========================================"
echo "Starting Standard Baseline Tests..."
echo "========================================"

# 5. MNIST + MLP + SGD (catastrophic forgetting baseline)
echo "[5/8] Starting MNIST + MLP + SGD baseline..."
tmux new-session -d -s val_baseline_sgd "cd /home/fneubuerger/CFC_Continual_Learning/mammoth && \
source ../.venv/bin/activate && \
python utils/main.py \
  --dataset seq-mnist \
  --model sgd \
  --backbone mnistmlp \
  --n_epochs 1 \
  --batch_size 32 \
  --lr 0.03 \
  --num_workers 0 \
  --seed 0 \
  2>&1 | tee ../validation_results/mnist_mlp_sgd.log; \
echo 'MNIST SGD baseline completed' && \
read -p 'Press enter to close...'"

# 6. MNIST + MLP + ER
echo "[6/8] Starting MNIST + MLP + ER baseline..."
tmux new-session -d -s val_baseline_er "cd /home/fneubuerger/CFC_Continual_Learning/mammoth && \
source ../.venv/bin/activate && \
python utils/main.py \
  --dataset seq-mnist \
  --model er \
  --backbone mnistmlp \
  --n_epochs 1 \
  --batch_size 32 \
  --lr 0.03 \
  --buffer_size 200 \
  --num_workers 0 \
  --seed 0 \
  2>&1 | tee ../validation_results/mnist_mlp_er.log; \
echo 'MNIST ER baseline completed' && \
read -p 'Press enter to close...'"

# 7. MNIST + MLP + DER++
echo "[7/8] Starting MNIST + MLP + DER++ baseline..."
tmux new-session -d -s val_baseline_derpp "cd /home/fneubuerger/CFC_Continual_Learning/mammoth && \
source ../.venv/bin/activate && \
python utils/main.py \
  --dataset seq-mnist \
  --model derpp \
  --backbone mnistmlp \
  --n_epochs 1 \
  --batch_size 32 \
  --lr 0.03 \
  --buffer_size 200 \
  --alpha 0.1 \
  --beta 0.5 \
  --num_workers 0 \
  --seed 0 \
  2>&1 | tee ../validation_results/mnist_mlp_derpp.log; \
echo 'MNIST DER++ baseline completed' && \
read -p 'Press enter to close...'"

# 8. MNIST + MLP + ER-ACE
echo "[8/8] Starting MNIST + MLP + ER-ACE baseline..."
tmux new-session -d -s val_baseline_ace "cd /home/fneubuerger/CFC_Continual_Learning/mammoth && \
source ../.venv/bin/activate && \
python utils/main.py \
  --dataset seq-mnist \
  --model er-ace \
  --backbone mnistmlp \
  --n_epochs 1 \
  --batch_size 32 \
  --lr 0.03 \
  --buffer_size 200 \
  --num_workers 0 \
  --seed 0 \
  2>&1 | tee ../validation_results/mnist_mlp_erace.log; \
echo 'MNIST ER-ACE baseline completed' && \
read -p 'Press enter to close...'"

echo ""
echo "=========================================================="
echo "All validation sessions started!"
echo "=========================================================="
echo ""
echo "Active tmux sessions:"
tmux ls
echo ""
echo "CfC Backbone Tests:"
echo "  View MNIST-CfC:      tmux attach -t val_mnist"
echo "  View CIFAR-CfC:      tmux attach -t val_cifar"
echo "  View TEP-CfC:        tmux attach -t val_tep_cfc"
echo "  View TEP-LSTM:       tmux attach -t val_tep_lstm"
echo ""
echo "Standard Baseline Tests:"
echo "  View SGD:            tmux attach -t val_baseline_sgd"
echo "  View ER:             tmux attach -t val_baseline_er"
echo "  View DER++:          tmux attach -t val_baseline_derpp"
echo "  View ER-ACE:         tmux attach -t val_baseline_ace"
echo ""
echo "Other Commands:"
echo "  List sessions:       tmux ls"
echo "  Detach:              Ctrl+b then d"
echo ""
echo "Results will be saved to: validation_results/"
echo ""
echo "Monitor progress:"
echo "  ./check_validation_results.sh"
echo ""
