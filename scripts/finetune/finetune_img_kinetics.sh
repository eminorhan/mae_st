#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=8:00:00
#SBATCH --job-name=finetune_maest_img_kinetics
#SBATCH --output=finetune_maest_img_kinetics_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

SHOT=10

# vit-h/14 sampling rate 8
srun python -u ../../finetune_img.py \
    --train_dir /vast/eo41/data/kinetics700-${SHOT}shot/train \
    --val_dir /scratch/eo41/data-video/kinetics700/val \
    --datafile_dir ../../datafiles/kinetics-${SHOT}shot \
    --num_classes 700 \
    --save_prefix "img_kinetics-${SHOT}shot" \
    --output_dir ../../models_finetuned \
    --model vit_huge_patch14 \
    --finetune /scratch/eo41/mae/models_vith14/s_vith14_checkpoint.pth \
    --batch_size_per_gpu 4 \
    --accum_iter 1 \
    --epochs 100000 \
    --num_frames 16 \
    --input_size 224 \
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