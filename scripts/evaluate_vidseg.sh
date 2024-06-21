#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=03:00:00
#SBATCH --job-name=evaluate_vidseg
#SBATCH --output=evaluate_vidseg_%A_%a.out
#SBATCH --array=0

srun python -u ../evaluate_vidseg.py \
	--data_path /vast/eo41/data/davis-2017/DAVIS \
	--output_dir ../vidseg/sayavakepicutego4d_448 \
	--ckpt ../models_finetuned_imagenet/new/sayavakepicutego4d_448_imagenet_0.02.pth \
	--img_size 448 \
	--save_prefix sayavakepicutego4d_448

echo "Done"
