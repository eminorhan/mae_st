#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:05:00
#SBATCH --job-name=eval_loss_maest
#SBATCH --output=eval_loss_maest_%A_%a.out
#SBATCH --array=0

python -u evaluate_loss.py \
    --model_arch mae_vit_huge_patch14 \
    --model_path "../models_maest/s_vith14_224_8_1_16_normpixloss_m09_noamp_Adam0001.pth" \
    --clip_dir "../demo" \
    --output_dir "../outputs" \
    --savefile_name "adept_demo" \
    --mask_ratio 0.5 \
    --mask_type "temporal"

echo "Done"