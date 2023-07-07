#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=8:00:00
#SBATCH --job-name=finetune_maest
#SBATCH --output=finetune_maest_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# # vit-h/14
# srun python -u /scratch/eo41/mae_st/main_finetune.py \
#     --train_dir /scratch/eo41/data-video/kinetics700-10shot \
#     --val_dir /scratch/eo41/data-video/kinetics700/val \
#     --save_prefix "s_vith14_224_4_1_16_normpixloss_m09_noamp_Adam0001_kinetics700_10shot_ft" \
#     --output_dir /scratch/eo41/mae_st/say_maest_ft \
#     --model vit_huge_patch14 \
#     --finetune "/scratch/eo41/mae_st/say_maest/s_vith14_224_4_1_16_normpixloss_m09_noamp_Adam0001.pth" \
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
#     --lr 0.0001 \
#     --clip_grad 1.0

# vit-b/14
srun python -u /scratch/eo41/mae_st/main_finetune.py \
    --train_dir /scratch/eo41/data-video/kinetics700-10shot \
    --val_dir /scratch/eo41/data-video/kinetics700/val \
    --save_prefix "y_vitb14_224_4_1_16_pixloss_m05_noamp_Adam0001_kinetics700_10shot_ft" \
    --output_dir /scratch/eo41/mae_st/say_maest_ft \
    --model vit_base_patch14 \
    --finetune "/scratch/eo41/mae_st/say_vitb14/y_vitb14_224_4_nopixnorm_m05.pth" \
    --batch_size_per_gpu 4 \
    --accum_iter 1 \
    --epochs 100000 \
    --num_frames 16 \
    --input_size 224 \
    --pin_mem \
    --num_workers 16 \
    --t_patch_size 2 \
    --repeat_aug 1 \
    --sampling_rate 4 \
    --lr 0.0001 \
    --clip_grad 1.0

echo "Done"