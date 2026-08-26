#!/bin/bash
# Download the Steel Plates Faults dataset (UCI ML Repository, id=198).
#
# Source: Buscema, M., Terzi, S., & Tastle, W. (2010). Steel Plates Faults
#         [Dataset]. UCI Machine Learning Repository.
#         https://doi.org/10.24432/C5J88N
# License: CC BY 4.0
set -euo pipefail

TARGET_DIR="data/SteelPlatesFaults"
URL="https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Downloading Steel Plates Faults dataset from UCI ML Repository..."
wget -c "$URL" -O steel_plates_faults.zip

echo "Extracting..."
unzip -o steel_plates_faults.zip

echo "Done. Files in $TARGET_DIR:"
ls -la

# mammoth's base_path() resolves relative to mammoth/, so mirror the data
# there too (matches the existing data/TEP convention in this repo).
cd - > /dev/null
mkdir -p mammoth/data/SteelPlatesFaults
cp "$TARGET_DIR"/Faults.NNA "$TARGET_DIR"/Faults27x7_var mammoth/data/SteelPlatesFaults/
echo "Mirrored dataset files into mammoth/data/SteelPlatesFaults/"
