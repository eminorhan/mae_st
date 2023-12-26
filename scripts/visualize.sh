#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:10:00
#SBATCH --job-name=visualize_maest
#SBATCH --output=visualize_maest_%A_%a.out
#SBATCH --array=0

python -u ../visualize.py \
    --model_arch mae_vit_huge_patch14 \
    --model_path ../models/s/s_vith14_224_8_1_16_pixloss_m09_accum1_Adam0001_250ep.pth \
    --video_dir /scratch/eo41/data-video/minute/S \
    --num_vids 50 \
    --mask_ratio 0.25

echo "Done"