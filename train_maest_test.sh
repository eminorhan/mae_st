#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=12:00:00
#SBATCH --job-name=train_maest_test
#SBATCH --output=train_maest_test_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# vit-b/14
srun python -u /scratch/eo41/mae_st/main_pretrain.py \
    --path_to_data_dir /scratch/eo41/data-video/minute/Y \
    --save_prefix "y_vitb14_224_4" \
    --output_dir /scratch/eo41/mae_st/say_vitb14 \
    --model mae_vit_base_patch14 \
    --resume "/scratch/eo41/mae_st/say_vitb14/y_vitb14_224_4.pth" \
    --batch_size_per_gpu 4 \
    --accum_iter 1 \
    --epochs 1000 \
    --num_frames 16 \
    --input_size 224 \
    --decoder_embed_dim 512 \
    --decoder_depth 4 \
    --pin_mem \
    --num_workers 16 \
    --t_patch_size 2 \
    --repeat_aug 4 \
    --sampling_rate 4 \
    --norm_pix_loss \
    --lr 0.0001 \
    --mask_ratio 0.7 \
    --pred_t_dim 8 \
    --clip_grad 1.0

echo "Done"