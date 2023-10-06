#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=48:00:00
#SBATCH --job-name=finetune_maest
#SBATCH --output=finetune_maest_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

# # vit-h/14 sampling rate 4
# srun python -u ../finetune.py \
#     --train_dir /scratch/eo41/data-video/kinetics700-50shot \
#     --val_dir /scratch/eo41/data-video/kinetics700/val \
#     --num_classes 700 \
#     --save_prefix "s_vith14_224_4_1_16_normpixloss_m09_Adam0001_k700_50shot_ft" \
#     --output_dir ../models_maest_ft \
#     --model vit_huge_patch14 \
#     --finetune "../models_maest/s_vith14_224_4_1_16_normpixloss_m09_noamp_Adam0001.pth" \
#     --batch_size_per_gpu 4 \
#     --accum_iter 1 \
#     --epochs 100000 \
#     --num_frames 16 \
#     --input_size 224 \
#     --pin_mem \
#     --num_workers 16 \
#     --t_patch_size 2 \
#     --repeat_aug 1 \
#     --sampling_rate 4 \
#     --blr 0.0024 \
#     --clip_grad 5.0 \
#     --rand_aug \
#     --mixup 0.8 \
#     --cutmix 1.0 \
#     --mixup_prob 1.0

# # vit-h/14 sampling rate 8
# srun python -u ../finetune.py \
#     --train_dir /scratch/eo41/data-video/kinetics700-50shot \
#     --val_dir /scratch/eo41/data-video/kinetics700/val \
#     --num_classes 700 \
#     --save_prefix "s_vith14_224_8_1_16_normpixloss_m09_noamp_Adam0001_k700_50shot_ft" \
#     --output_dir ../models_maest_ft \
#     --model vit_huge_patch14 \
#     --finetune "../models_maest/s_vith14_224_8_1_16_normpixloss_m09_noamp_Adam0001.pth" \
#     --batch_size_per_gpu 4 \
#     --accum_iter 1 \
#     --epochs 100000 \
#     --num_frames 16 \
#     --input_size 224 \
#     --pin_mem \
#     --num_workers 16 \
#     --t_patch_size 2 \
#     --repeat_aug 1 \
#     --sampling_rate 8 \
#     --blr 0.0024 \
#     --clip_grad 5.0 \
#     --rand_aug \
#     --mixup 0.8 \
#     --cutmix 1.0 \
#     --mixup_prob 1.0

# vit-h/14 sampling rate 16
srun python -u ../finetune.py \
    --train_dir /scratch/eo41/data-video/kinetics700-50shot \
    --val_dir /scratch/eo41/data-video/kinetics700/val \
    --num_classes 700 \
    --save_prefix "s_vith14_224_16_1_16_normpixloss_m09_Adam0001_k700_50shot_ft" \
    --output_dir ../say_maest_ft \
    --model vit_huge_patch14 \
    --finetune "../models_maest/s_vith14_224_16_1_16_normpixloss_m09_noamp_Adam0001.pth" \
    --batch_size_per_gpu 4 \
    --accum_iter 1 \
    --epochs 100000 \
    --num_frames 16 \
    --input_size 224 \
    --pin_mem \
    --num_workers 8 \
    --t_patch_size 2 \
    --repeat_aug 1 \
    --sampling_rate 16 \
    --blr 0.0024 \
    --clip_grad 5.0 \
    --rand_aug \
    --mixup 0.8 \
    --cutmix 1.0 \
    --mixup_prob 1.0

echo "Done"