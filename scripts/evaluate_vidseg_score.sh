#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=00:10:00
#SBATCH --job-name=evaluate_vidseg_score
#SBATCH --output=evaluate_vidseg_score_%A_%a.out
#SBATCH --array=0

srun python -u /scratch/eo41/davis2017-evaluation/evaluation_method.py \
	--task semi-supervised \
	--results_path ../vidseg/sayavakepicutego4d_448 \
	--davis_path /vast/eo41/data/davis-2017/DAVIS

echo "Done"