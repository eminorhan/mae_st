#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=01:00:00
#SBATCH --job-name=upload_models
#SBATCH --output=upload_models_%A_%a.out

python -u ../upload_models.py 

echo "Done"