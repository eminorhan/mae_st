#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=02:00:00
#SBATCH --job-name=embed_s_ssv2
#SBATCH --output=embed_s_ssv2_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# s - ssv2
srun python -u ../embed.py \
    --data_dirs /vast/eo41/ssv2/val \
    --save_prefix "s_ssv2_0-shot" \
    --output_dir ../embeddings \
    --datafile_dir ../datafiles/ssv2 \
    --model vit_huge_patch14 \
    --resume ../models/s/s_vith14_224_8_1_16_pixloss_m09_accum1_Adam0001_250ep.pth \
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