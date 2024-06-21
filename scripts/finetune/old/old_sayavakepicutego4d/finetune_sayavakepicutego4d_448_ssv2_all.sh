#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=48:00:00
#SBATCH --job-name=finetune_maest_sayavakepicutego4d_448_ssv2_all
#SBATCH --output=finetune_maest_sayavakepicutego4d_448_ssv2_all_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

# vit-h/14 sampling rate 8
srun python -u ../../finetune.py \
    --train_dir /vast/eo41/ssv2/train \
    --val_dir /vast/eo41/ssv2/val \
    --datafile_dir ../../datafiles/ssv2-all \
    --num_classes 174 \
    --save_prefix "sayavakepicutego4d_448_ssv2-all" \
    --output_dir ../../models_finetuned \
    --model vit_huge_patch14 \
    --resume ../../models_finetuned/sayavakepicutego4d_448_ssv2-all.pth \
    --batch_size_per_gpu 4 \
    --accum_iter 4 \
    --epochs 100000 \
    --num_frames 16 \
    --img_size 448 \
    --pin_mem \
    --num_workers 16 \
    --t_patch_size 2 \
    --repeat_aug 1 \
    --sampling_rate 8 \
    --blr 0.0012 \
    --clip_grad 5.0 \
    --mixup 0 \
    --cutmix 0.0 \
    --train_jitter_scales 448 448

echo "Done"
