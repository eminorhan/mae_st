#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=11:00:00
#SBATCH --job-name=train_maest_sayavakepicutego4d
#SBATCH --output=train_maest_sayavakepicutego4d_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

# vit-h/14 sayavakepicutego4d
srun python -u ../../pretrain.py \
    --data_dirs /scratch/projects/lakelab/data_videos \
    --datafile_dir ../../datafiles/new/sayavakepicutego4d \
    --save_prefix sayavakepicutego4d \
    --output_dir ../../models/new/sayavakepicutego4d \
    --model mae_vit_huge_patch14 \
    --resume ../../models/sayavakepicutego4d/sayavakepicutego4d_vith14_224_8_1_16_pixloss_m09_accum1_Adam0001.pth \
    --batch_size_per_gpu 4 \
    --accum_iter 1 \
    --epochs 100000 \
    --num_frames 16 \
    --img_size 224 \
    --decoder_embed_dim 512 \
    --decoder_depth 4 \
    --pin_mem \
    --num_workers 16 \
    --t_patch_size 2 \
    --repeat_aug 16 \
    --sampling_rate 8 \
    --lr 0.00001 \
    --weight_decay 0.05 \
    --mask_ratio 0.9 \
    --pred_t_dim 16 \
    --clip_grad 0.1

echo "Done"