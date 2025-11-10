#!/bin/bash
# Compare different continual learning methods on seq-mnist

cd mammoth

echo "========================================="
echo "Running SGD (no continual learning)"
echo "========================================="
python utils/main.py --dataset seq-mnist --model sgd --lr 0.03 --n_epochs 5 --batch_size 32 --nowand 1

echo ""
echo "========================================="
echo "Running EWC (Elastic Weight Consolidation)"
echo "========================================="
python utils/main.py --dataset seq-mnist --model ewc_on --lr 0.03 --n_epochs 5 --batch_size 32 --nowand 1

echo ""
echo "========================================="
echo "Running ER (Experience Replay)"
echo "========================================="
python utils/main.py --dataset seq-mnist --model er --lr 0.03 --n_epochs 5 --batch_size 32 --buffer_size 200 --nowand 1

echo ""
echo "All experiments completed!"
