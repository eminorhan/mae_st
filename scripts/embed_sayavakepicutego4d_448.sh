#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240GB
#SBATCH --time=09:00:00
#SBATCH --job-name=embed_sayavakepicutego4d_448_ssv2
#SBATCH --output=embed_sayavakepicutego4d_448_ssv2_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

    # --resume ../models/new/sayavakepicutego4d_448/sayavakepicutego4d_448.pth \

# hvm1 - 448 - ssv2
srun python -u ../embed.py \
    --data_dirs /vast/eo41/ssv2/val \
    --save_prefix "hvm1_448_ssv2_10-shot" \
    --output_dir ../embeddings \
    --datafile_dir ../datafiles/old/ssv2 \
    --model vit_huge_patch14 \
    --resume ../models_finetuned/new/sayavakepicutego4d_448_ssv2-10shot.pth \
    --batch_size_per_gpu 128 \
    --num_frames 16 \
    --img_size 448 \
    --pin_mem \
    --num_workers 8 \
    --t_patch_size 2 \
    --repeat_aug 1 \
    --sampling_rate 8 \
    --train_jitter_scales 448 448 \
    --eval

echo "Done"