#!/bin/bash
#SBATCH --job-name=smiles-training
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=your_partition  # specify if your cluster requires it

# Activate the Conda environment
source activate mamba

# Run the Python script, adjust the path to where your script is located
python mamba_gridsearch.py