#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=00:05:00
#SBATCH --job-name=train_maest_test
#SBATCH --output=train_maest_test_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# vit-b/14
srun python -u /scratch/eo41/mae_st/main_pretrain.py \
    --path_to_data_dir /scratch/eo41/data-video/minute/Ytest2 \
    --save_prefix "ytest2_vitb14_224_4_nopixnorm_m05" \
    --output_dir /scratch/eo41/mae_st/say_vitb14 \
    --model mae_vit_base_patch14 \
    --resume "/scratch/eo41/mae_st/say_vitb14/ytest2_vitb14_224_4_nopixnorm_m05.pth" \
    --batch_size_per_gpu 1 \
    --accum_iter 1 \
    --epochs 100000 \
    --num_frames 16 \
    --input_size 224 \
    --decoder_embed_dim 512 \
    --decoder_depth 4 \
    --pin_mem \
    --num_workers 16 \
    --t_patch_size 2 \
    --repeat_aug 16 \
    --sampling_rate 4 \
    --lr 0.00005 \
    --mask_ratio 0.5 \
    --pred_t_dim 8 \
    --clip_grad 0.1

echo "Done"