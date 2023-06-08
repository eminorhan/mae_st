#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=00:10:00
#SBATCH --job-name=visualize_maest
#SBATCH --output=visualize_maest_%A_%a.out
#SBATCH --array=0

python -u visualize.py

echo "Done"