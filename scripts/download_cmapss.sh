#!/bin/bash
# Download the NASA C-MAPSS Turbofan Engine Degradation Simulation dataset.
#
# Source: Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage
#         Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.
#         In Proceedings of the 1st International Conference on Prognostics
#         and Health Management (PHM08), Denver, CO.
# Host:   NASA Open Data Portal (Prognostics Center of Excellence)
#         https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
# License: Public domain (US Government work); no separate license specified
#          by NASA on the dataset page.
#
# NOTE: this dataset is natively a REGRESSION benchmark (predict Remaining
# Useful Life in operating cycles). It is not yet wired into a mammoth
# ContinualDataset -- see docs/industrial_benchmarks.md for the open design
# question (how to bin RUL into classes for a Class-IL split) before use.
set -euo pipefail

TARGET_DIR="data/CMAPSS"
URL="https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Downloading NASA C-MAPSS turbofan degradation dataset..."
wget -c "$URL" -O CMAPSSData.zip

echo "Extracting..."
unzip -o CMAPSSData.zip

echo "Done. Files in $TARGET_DIR:"
ls -la

cd - > /dev/null
mkdir -p mammoth/data/CMAPSS
cp "$TARGET_DIR"/train_FD00?.txt "$TARGET_DIR"/test_FD00?.txt "$TARGET_DIR"/RUL_FD00?.txt mammoth/data/CMAPSS/
echo "Mirrored dataset files into mammoth/data/CMAPSS/"
