#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=492GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train_maest
#SBATCH --output=train_maest_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

# vit-h/14
srun python -u /scratch/eo41/mae_st/main_pretrain.py \
    --path_to_data_dir /scratch/eo41/data-video/minute/Y \
    --save_prefix "y_vith14_224_4" \
    --output_dir /scratch/eo41/mae_st/say_vith14 \
    --model mae_vit_huge_patch14 \
    --resume "" \
    --batch_size_per_gpu 1 \
    --epochs 1000 \
    --num_frames 16 \
    --input_size 224 \
    --decoder_embed_dim 512 \
    --decoder_depth 4 \
    --pin_mem \
    --num_workers 4 \
    --t_patch_size 2 \
    --repeat_aug 23 \
    --sampling_rate 4 \
    --norm_pix_loss \
    --lr 0.0001 \
    --mask_ratio 0.9 \
    --pred_t_dim 8 \
    --clip_grad 0.02

echo "Done"