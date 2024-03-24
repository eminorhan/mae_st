#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=12:00:00
#SBATCH --job-name=finetune_maest_sayavakepicutego4d_ssv2
#SBATCH --output=finetune_maest_sayavakepicutego4d_ssv2_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

SHOT=50

# vit-h/14 sampling rate 8
srun python -u ../../finetune.py \
    --train_dir /vast/eo41/ssv2/train_${SHOT}shot/train \
    --val_dir /vast/eo41/ssv2/val \
    --datafile_dir ../../datafiles/ssv2-${SHOT}shot \
    --num_classes 174 \
    --save_prefix "sayavakepicutego4d_ssv2-${SHOT}shot" \
    --output_dir ../../models_finetuned \
    --model vit_huge_patch14 \
    --finetune ../../models/sayavakepicutego4d/sayavakepicutego4d_vith14_224_8_1_16_pixloss_m09_accum1_Adam0001.pth \
    --batch_size_per_gpu 16 \
    --accum_iter 2 \
    --epochs 100000 \
    --num_frames 16 \
    --img_size 224 \
    --pin_mem \
    --num_workers 16 \
    --t_patch_size 2 \
    --repeat_aug 1 \
    --sampling_rate 8 \
    --blr 0.0024 \
    --clip_grad 5.0 \
    --mixup 0 \
    --cutmix 0.0

echo "Done"