#!/bin/bash
cd ./mammoth
source ../.venv/bin/activate
python utils/main.py --dataset tennessee-eastman --model ewc_on --backbone tepcfc --n_epochs 1 --batch_size 64 --num_workers 0 --savecheck task --enable_other_metrics 1 --lr 0.001 --e_lambda 1000 --gamma 1.0 --num_features 52 --num_classes 22 --hidden_size 128 --use_ncp_wiring 1 --seed 0
