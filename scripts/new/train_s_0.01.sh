#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=1:00:00
#SBATCH --job-name=train_maest_s_0.01
#SBATCH --output=train_maest_s_0.01_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# vit-h/14 sayavakepicutego4d
srun python -u ../../pretrain.py \
    --data_dirs /scratch/projects/lakelab/data_videos/saycam_s \
    --datafile_dir ../../datafiles/new/s_0.01 \
    --data_frac 0.01 \
    --save_prefix s_0.01 \
    --output_dir ../../models/new/s_0.01 \
    --model mae_vit_huge_patch14 \
    --resume ../../models/old/s-0.01/s_0.01_vith14_224_8_1_16_pixloss_m09_accum1_Adam0001.pth \
    --batch_size_per_gpu 4 \
    --accum_iter 1 \
    --epochs 20 \
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