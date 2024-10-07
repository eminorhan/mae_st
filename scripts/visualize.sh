#!/bin/bash

#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=00:30:00
#SBATCH --job-name=visualize_maest
#SBATCH --output=visualize_maest_%A_%a.out
#SBATCH --array=0

python -u ../visualize.py \
    --model_arch mae_vit_huge_patch14 \
    --model_path ../models/new/s/s.pth \
    --video_dir /scratch/eo41/data-video/minute/Y \
    --num_vids 512 \
    --mask_ratio 0.25

echo "Done"