#!/bin/bash
#SBATCH -A m4539
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -J nn_uq_test
#SBATCH -o nn_uq_%j.out
#SBATCH -e nn_uq_%j.err

# Load modules
module load pytorch/2.8.0

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install numpy scipy scikit-learn matplotlib
else
    source venv/bin/activate
fi

# Run the main script
python main.py
