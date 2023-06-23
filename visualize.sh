#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=00:01:00
#SBATCH --job-name=visualize_maest
#SBATCH --output=visualize_maest_%A_%a.out
#SBATCH --array=0

python -u visualize.py \
    --model_arch mae_vit_base_patch14 \
    --model_path "kinetics700_vitb14/kinetics700_10shot_vitb14_224_4.pth" \
    --eval_clip_name "demo1" \
    --mask_ratio 0.7

echo "Done"