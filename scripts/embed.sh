#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=01:00:00
#SBATCH --job-name=embed_maest
#SBATCH --output=embed_maest_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# # vit-h/14 saycam
# srun python -u ../embed.py \
#     --data_dirs /vast/eo41/ssv2/data \
#     --save_prefix "s_ssv2" \
#     --output_dir ../embeddings \
#     --datafile_dir ../datafiles/ssv2 \
#     --model vit_huge_patch14 \
#     --resume ../models/saycam/s_vith14_224_8_1_16_pixloss_m09_accum1_Adam0001_80ep.pth \
#     --batch_size_per_gpu 100 \
#     --num_frames 16 \
#     --input_size 224 \
#     --pin_mem \
#     --num_workers 8 \
#     --t_patch_size 2 \
#     --repeat_aug 1 \
#     --sampling_rate 8 \
#     --eval

# vit-h/14 saycam
srun python -u ../embed.py \
    --data_dirs /vast/eo41/ssv2/data \
    --save_prefix "kinetics_ssv2" \
    --output_dir ../embeddings \
    --datafile_dir ../datafiles/ssv2 \
    --model vit_huge_patch14 \
    --resume ../models/kinetics/kinetics_vith14_224_8_1_16_pixloss_m09_accum1_Adam0001_22ep.pth \
    --batch_size_per_gpu 100 \
    --num_frames 16 \
    --input_size 224 \
    --pin_mem \
    --num_workers 8 \
    --t_patch_size 2 \
    --repeat_aug 1 \
    --sampling_rate 8 \
    --eval

echo "Done"