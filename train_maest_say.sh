#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300GB
#SBATCH --time=12:00:00
#SBATCH --job-name=train_maest_say
#SBATCH --output=train_maest_say_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=2

# vit-b/16
srun python -u /scratch/eo41/mae_st/run_pretrain.py \
    --path_to_data_dir /vast/eo41/data_video/SAY \
    --output_dir /scratch/eo41/mae_st/say_vitb16 \
    --model mae_vit_base_patch16 \
    --save_prefix "say_vitb16" \
    --resume "" \
    --batch_size_per_gpu 1 \
    --epochs 100 \
    --num_frames 16 \
    --decoder_embed_dim 512 \
    --decoder_depth 4 \
    --pin_mem \
    --num_workers 4 \
    --t_patch_size 2 \
    --repeat_aug 56 \
    --sampling_rate 4 \
    --norm_pix_loss \
    --lr 0.0001 \
    --warmup_epochs 5 \
    --mask_ratio 0.9 \
    --pred_t_dim 8 \
    --clip_grad 0.02

echo "Done"