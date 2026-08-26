#!/bin/bash
# Download the SECOM dataset (UCI ML Repository, id=179).
#
# Source: McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine
#         Learning Repository. https://doi.org/10.24432/C54305
# License: CC BY 4.0
set -euo pipefail

TARGET_DIR="data/SECOM"
URL="https://archive.ics.uci.edu/static/public/179/secom.zip"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Downloading SECOM dataset from UCI ML Repository..."
wget -c "$URL" -O secom.zip

echo "Extracting..."
unzip -o secom.zip

echo "Done. Files in $TARGET_DIR:"
ls -la

cd - > /dev/null
mkdir -p mammoth/data/SECOM
cp "$TARGET_DIR"/secom.data "$TARGET_DIR"/secom_labels.data mammoth/data/SECOM/
echo "Mirrored dataset files into mammoth/data/SECOM/"
