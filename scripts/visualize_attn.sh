#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:10:00
#SBATCH --job-name=visualize_attn
#SBATCH --output=visualize_attn_%A_%a.out
#SBATCH --array=0

python -u ../visualize_attn.py \
    --model vit_huge_patch14 \
    --model_path ../models/s/s_vith14_224_8_1_16_pixloss_m09_accum1_Adam0001_250ep.pth \
    --video_dir /scratch/eo41/data-video/minute/S \
    --num_vids 100 \
    --num_frames 16 \
    --input_size 224 \
    --pin_mem \
    --num_workers 8 \
    --t_patch_size 2 \
    --repeat_aug 1 \
    --sampling_rate 8

echo "Done"