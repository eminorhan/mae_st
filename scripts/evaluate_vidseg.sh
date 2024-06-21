#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=01:10:00
#SBATCH --job-name=evaluate_vidseg
#SBATCH --output=evaluate_vidseg_%A_%a.out
#SBATCH --array=0

srun python -u ../evaluate_vidseg.py \
	--data_path /vast/eo41/data/davis-2017/DAVIS \
	--output_dir ../vidseg/kinetics \
	--ckpt ../models_finetuned_imagenet/new/kinetics/kinetics_imagenet_0.02.pth \
	--img_size 224 \
	--save_prefix kinetics

echo "Done"
