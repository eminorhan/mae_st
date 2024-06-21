#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=48:00:00
#SBATCH --job-name=finetune_kinetics_imagenet
#SBATCH --output=finetune_kinetics_imagenet_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

# vit-h/14 sampling rate 8
srun python -u ../../finetune_on_image.py \
    --train_data_path /scratch/work/public/imagenet/train \
    --val_data_path /scratch/eo41/imagenet/val \
    --frac_retained 0.02 \
    --num_classes 1000 \
    --save_prefix "kinetics_imagenet_0.02" \
    --output_dir ../../models_finetuned_imagenet \
    --model vit_huge_patch14 \
    --resume ../../models/new/kinetics/kinetics.pth \
    --batch_size_per_gpu 16 \
    --accum_iter 1 \
    --epochs 100000 \
    --num_frames 16 \
    --img_size 224 \
    --pin_mem \
    --num_workers 16 \
    --t_patch_size 2 \
    --blr 0.0012 \
    --clip_grad 5.0 \
    --mixup 0 \
    --cutmix 0.0

echo "Done"