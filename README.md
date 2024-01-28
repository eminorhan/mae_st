## Masked Autoencoders As Spatiotemporal Learners

This is my personal copy of Facebook's [Spatiotemporal Masked Autoencoders](https://github.com/facebookresearch/mae_st) repository customized for my own purposes. The code here can be used to train and finetune spatiotemporal MAEs.

### Usage examples

* **Training:** To train a spatiotemporal MAE model with a ViT-H/14 architecture from scratch on your data, use [`pretrain.py`](https://github.com/eminorhan/mae_st/blob/master/pretrain.py), *e.g.*:
```python
python -u pretrain.py \
    --data_dirs DATA_DIRS \
    --datafile_dir DATAFILE_DIR \
    --save_prefix INFORMATIVE_SAVE_PREFIX \
    --output_dir OUTPUT_DIR \
    --model 'mae_vit_huge_patch14' \
    --batch_size_per_gpu 1 \
    --accum_iter 1 \
    --epochs 100000 \
    --num_frames 16 \
    --img_size 224 \
    --decoder_embed_dim 512 \
    --decoder_depth 4 \
    --pin_mem \
    --t_patch_size 2 \
    --repeat_aug 16 \
    --sampling_rate 8 \
    --lr 0.0001 \
    --weight_decay 0.05 \
    --mask_ratio 0.9 \
    --pred_t_dim 16 \
    --clip_grad 0.1
```
Here, `DATA_DIRS` is a list of directories containing the video files, `DATAFILE_DIR` is the directory where a `.csv` file containing all the training video file paths (optionally, with the corresponding class labels) will be saved, and `OUTPUT_DIR` is the directory where the checkpoints and training logs will be saved.

* **Finetuning on videos:** To finetune a ViT-H/14 model on a downstream video recognition task, use [`finetune.py`](https://github.com/eminorhan/mae_st/blob/master/finetune.py), *e.g.*:
```python
python -u finetune.py \
    --train_dir TRAIN_DIR \
    --val_dir VAL_DIR \
    --datafile_dir DATAFILE_DIR \
    --save_prefix INFORMATIVE_SAVE_PREFIX \
    --output_dir OUTPUT_DIR \
    --finetune SPATIOTEMPORAL_MAE_CHECKPOINT \
    --num_classes 174 \
    --model 'vit_huge_patch14' \
    --batch_size_per_gpu 4 \
    --accum_iter 1 \
    --epochs 100000 \
    --num_frames 16 \
    --input_size 224 \
    --pin_mem \
    --t_patch_size 2 \
    --repeat_aug 1 \
    --sampling_rate 8 \
    --blr 0.0024 \
    --clip_grad 5.0 \
    --mixup 0 \
    --cutmix 0.0
```
Here, `TRAIN_DIR` and `VAL_DIR` are the directories containing the training and validation videos, respectively, and `SPATIOTEMPORAL_MAE_CHECKPOINT` is the path to the spatiotemporal MAE checkpoint the model is initialized with (use `""` here if you would like to finetune the model from scratch without any pretraining).
