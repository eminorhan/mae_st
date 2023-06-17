#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=00:05:00
#SBATCH --job-name=visualize_maest
#SBATCH --output=visualize_maest_%A_%a.out
#SBATCH --array=0

python -u visualize.py \
    --model_arch mae_vit_base_patch14 \
    --model_path "say_vitb14/y_vitb14_224_4.pth" \
    --eval_clip_name "demo4" \
    --mask_ratio 0.75

echo "Done"