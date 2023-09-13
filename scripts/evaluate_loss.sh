#!/bin/bash

#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=00:50:00
#SBATCH --job-name=eval_loss_maest
#SBATCH --output=eval_loss_maest_%A_%a.out
#SBATCH --array=0

# python -u evaluate_loss.py \
#     --model_arch mae_vit_huge_patch14 \
#     --model_path "../models_maest/s_vith14_224_8_1_16_normpixloss_m09_noamp_Adam0001.pth" \
#     --clip_dir "../demo" \
#     --output_dir "../outputs" \
#     --savefile_name "adept_demo" \
#     --mask_ratio 0.5 \
#     --mask_type "temporal"

python -u ../evaluate_loss.py \
    --model_arch mae_vit_base_patch14 \
    --model_path "../models_maest/adept_vitb14_224_sampling8_bs16_repeat2_pixloss_m09_accum1_Adam0001_predtdim16.pth" \
    --clip_dir "/scratch/eo41/adept/videos/test" \
    --output_dir "../outputs" \
    --savefile_name "adept_demo_samples5_mask025" \
    --mask_ratio 0.25 \
    --mask_type "temporal" \
    --num_samples 5

echo "Done"