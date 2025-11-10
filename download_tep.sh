#!/bin/bash
# Download Tennessee Eastman Process dataset

echo "Downloading Tennessee Eastman Process dataset..."

# Create data directory
mkdir -p data/TEP
cd data/TEP

# Download TEP dataset from common source
# Note: TEP dataset is available from multiple sources. Using a reliable mirror.

echo "Downloading training data..."
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d00.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d01.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d02.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d03.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d04.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d05.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d06.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d07.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d08.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d09.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d10.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d11.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d12.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d13.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d14.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d15.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d16.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d17.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d18.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d19.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d20.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d21.dat"

echo "Downloading test data..."
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d00_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d01_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d02_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d03_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d04_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d05_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d06_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d07_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d08_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d09_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d10_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d11_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d12_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d13_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d14_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d15_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d16_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d17_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d18_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d19_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d20_te.dat"
wget -c "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/d21_te.dat"

echo "Download complete!"
echo "Dataset location: $(pwd)"
echo ""
echo "Dataset structure:"
echo "  - d00.dat to d21.dat: Training data (normal + 21 faults)"
echo "  - d00_te.dat to d21_te.dat: Test data"
echo "  - Each file has 500 samples × 52 variables"
