#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train_maest
#SBATCH --output=train_maest_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# vit-h/14
srun python -u ../main_pretrain.py \
    --path_to_data_dir /scratch/eo41/adept/videos/train \
    --save_prefix "adept_vitb14_224_sampling8_bs16_repeat2_pixloss_m09_accum1_Adam0001" \
    --output_dir ../models_maest \
    --model mae_vit_base_patch14 \
    --resume "" \
    --batch_size_per_gpu 16 \
    --accum_iter 1 \
    --epochs 100000 \
    --num_frames 16 \
    --input_size 224 \
    --decoder_embed_dim 512 \
    --decoder_depth 4 \
    --pin_mem \
    --num_workers 16 \
    --t_patch_size 2 \
    --repeat_aug 2 \
    --sampling_rate 8 \
    --lr 0.0001 \
    --weight_decay 0.05 \
    --mask_ratio 0.9 \
    --pred_t_dim 8 \
    --clip_grad 0.1

echo "Done"