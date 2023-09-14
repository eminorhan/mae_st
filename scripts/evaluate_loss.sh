#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=00:59:00
#SBATCH --job-name=eval_loss_maest
#SBATCH --output=eval_loss_maest_%A_%a.out
#SBATCH --array=0

# # trained on adept train
# python -u ../evaluate_loss.py \
#     --model_arch mae_vit_base_patch14 \
#     --model_path "../models_maest/adept_vitb14_224_sampling8_bs16_repeat2_pixloss_m09_accum1_Adam0001_predtdim16.pth" \
#     --clip_dir "/scratch/eo41/adept/videos/test" \
#     --output_dir "../outputs" \
#     --savefile_name "adept_demo_samples5_mask06_random" \
#     --mask_ratio 0.6 \
#     --mask_type "random" \
#     --num_samples 5

# # pretrained on saycam, finetuned on adept
# python -u ../evaluate_loss.py \
#     --model_arch mae_vit_huge_patch14 \
#     --model_path "../models_maest/s+adept_vith14_224_sampling8_bs16_repeat1_pixloss_m09_accum1_Adam0001_predtdim16.pth" \
#     --clip_dir "/scratch/eo41/adept/videos/test" \
#     --output_dir "../outputs" \
#     --savefile_name "s+adept_demo_samples5_mask07_random" \
#     --mask_ratio 0.7 \
#     --mask_type "random" \
#     --num_samples 5

# only trained on saycam
python -u ../evaluate_loss.py \
    --model_arch mae_vit_huge_patch14 \
    --model_path "../models_maest/s_vith14_224_8_1_16_normpixloss_m09_noamp_Adam0001.pth" \
    --clip_dir "/scratch/eo41/adept/videos/test" \
    --output_dir "../outputs" \
    --savefile_name "s_demo_samples5_mask07_random" \
    --mask_ratio 0.7 \
    --mask_type "random" \
    --num_samples 5

echo "Done"