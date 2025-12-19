#!/bin/bash
# Debug run for HOPE
source /home/fneubuerger/CFC_Continual_Learning/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

python mammoth/utils/main.py \
    --dataset seq-cifar10 \
    --model hope \
    --backbone hope \
    --lr 0.001 \
    --hope_lr 0.001 \
    --n_epochs 1 \
    --batch_size 32 \
    --input_size 32 \
    --input_channels 3 \
    --num_workers 0
