#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480GB
#SBATCH --time=48:00:00
#SBATCH --job-name=finetune_maest
#SBATCH --output=finetune_maest_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

# vit-h/14
srun python -u /scratch/eo41/mae_st/main_finetune.py \
    --train_dir /scratch/eo41/data-video/kinetics700-100shot \
    --val_dir /scratch/eo41/data-video/kinetics700/val \
    --num_classes 700 \
    --save_prefix "s_vith14_224_4_1_16_normpixloss_m09_noamp_Adam0001_k700_100shot_ft" \
    --output_dir /scratch/eo41/mae_st/say_maest_ft \
    --model vit_huge_patch14 \
    --finetune "/scratch/eo41/mae_st/say_maest/s_vith14_224_4_1_16_normpixloss_m09_noamp_Adam0001.pth" \
    --batch_size_per_gpu 4 \
    --accum_iter 1 \
    --epochs 100000 \
    --num_frames 16 \
    --input_size 224 \
    --pin_mem \
    --num_workers 8 \
    --t_patch_size 2 \
    --repeat_aug 1 \
    --sampling_rate 4 \
    --lr 0.00005 \
    --clip_grad 1.0 \
    --rand_aug \
    --mixup 0.8 \
    --cutmix 1.0 \
    --mixup_prob 1.0

# # scratch
# srun python -u /scratch/eo41/mae_st/main_finetune.py \
#     --train_dir /scratch/eo41/data-video/kmini/train \
#     --val_dir /scratch/eo41/data-video/kmini/val \
#     --num_classes 16 \
#     --save_prefix "scratch+kmini_50shot_Adam00001" \
#     --output_dir /scratch/eo41/mae_st/say_maest_ft \
#     --model vit_huge_patch14 \
#     --finetune "" \
#     --batch_size_per_gpu 4 \
#     --accum_iter 1 \
#     --epochs 100000 \
#     --num_frames 16 \
#     --input_size 224 \
#     --pin_mem \
#     --num_workers 8 \
#     --t_patch_size 2 \
#     --repeat_aug 1 \
#     --sampling_rate 4 \
#     --lr 0.00001 \
#     --clip_grad 1.0

echo "Done"