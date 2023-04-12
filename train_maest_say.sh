#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=1:00:00
#SBATCH --job-name=train_maest_say
#SBATCH --output=train_maest_say_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# vit-h/14
srun python -u /scratch/eo41/mae_st/run_pretrain.py \
    --path_to_data_dir /vast/eo41/data_video/SAY \
    --batch_size 2 \
    --model mae_vit_large_patch16 \
    --no_env \
    --epochs 100 \
    --distributed \
    --num_frames 16 \
    --decoder_embed_dim 512 \
    --decoder_depth 4 \
    --pin_mem \
    --num_workers 14 \
    --t_patch_size 2 \
    --repeat_aug 4 \
    --sampling_rate 4 \
    --norm_pix_loss \
    --blr 1.6e-3 \
    --warmup_epochs 5 \
    --mask_ratio 0.9 \
    --pred_t_dim 8 \
    --clip_grad 0.02 \

echo "Done"