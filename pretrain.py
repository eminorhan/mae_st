# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import argparse
import datetime
import json
import os
import time
import torch
import torch.backends.cudnn as cudnn
from iopath.common.file_io import g_pathmgr as pathmgr
from pathlib import Path

import util.misc as misc
import models_mae
from engine_pretrain import train_one_epoch
from util.kinetics import Kinetics
from util.misc import NativeScalerWithGradNormCount as NativeScaler

def get_args_parser():
    parser = argparse.ArgumentParser("Spatiotemporal MAE pre-training", add_help=False)
    parser.add_argument("--batch_size_per_gpu", default=4, type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--accum_iter", default=1, type=int, help="Accumulate gradient iterations")
    parser.add_argument("--save_prefix", default="", type=str, help="Prefix for saving checkpoint and log files")

    # Data args
    parser.add_argument("--data_dirs", type=str, default=[""], nargs="+", help="Data paths")
    parser.add_argument("--datafile_dir", type=str, default="./datafiles", help="Store data files here")
    parser.add_argument("--output_dir", default="./output_dir", help="Path where to save, empty for no saving")
    parser.add_argument("--data_frac", default=1.0, type=float, help="Fraction of data to be used for training")

    # Model parameters
    parser.add_argument("--model", default="mae_vit_large_patch16", type=str, help="Name of model to train")
    parser.add_argument("--img_size", default=224, type=int, help="Image size")
    parser.add_argument("--mask_ratio", default=0.9, type=float, help="Masking ratio (percentage of removed patches).")
    parser.add_argument("--norm_pix_loss", action="store_true", help="Use (per-patch) normalized pixels as targets for computing loss")
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    parser.set_defaults(norm_pix_loss=False)

    # Training related parameters
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (absolute lr)")
    parser.add_argument("--device", default="cuda", help="Device to use for training / testing")
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--start_epoch", default=0, type=int, help="Start epoch")
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    # Video related configs
    parser.add_argument("--decoder_embed_dim", default=512, type=int)
    parser.add_argument("--decoder_depth", default=8, type=int)
    parser.add_argument("--decoder_num_heads", default=16, type=int)
    parser.add_argument("--t_patch_size", default=2, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--sampling_rate", default=4, type=int)
    parser.add_argument("--repeat_aug", default=4, type=int)
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--bias_wd", action="store_true")
    parser.add_argument("--num_checkpoint_del", default=20, type=int)
    parser.add_argument("--trunc_init", action="store_true")
    parser.add_argument("--target_fps", default=30, type=int)
    parser.add_argument("--jitter_scales_relative", default=[0.5, 1.0], type=float, nargs="+")
    parser.add_argument("--jitter_aspect_relative", default=[0.75, 1.3333], type=float, nargs="+")
    parser.add_argument("--color_jitter", type=bool, default=False, help="Color augmentation during training")
    parser.add_argument("--beta", default=None, type=float, nargs="+")
    parser.add_argument("--pred_t_dim", type=int, default=16)
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument("--fp32", action="store_true")
    parser.set_defaults(fp32=True)

    return parser

def find_mp4_files(directories):
    """Recursively search for .mp4 files in directories and their subdirectories"""
    mp4_files = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            files = sorted(files)
            for file in files:
                if file.endswith((".mp4", ".MP4", ".mkv", ".webm")):
                    mp4_files.append((os.path.join(root, file), os.path.basename(root)))
    return mp4_files

def write_csv(video_files, save_dir, save_name):
    """Write the .csv file with video path and subfolder index"""
    with open(os.path.join(save_dir, f'{save_name}.csv'), 'w', newline='') as csvfile:
        for video_file, _ in video_files:
            csvfile.write(f"{video_file}, {1}\n")

def main(args):
    misc.init_distributed_mode(args)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))
    device = torch.device(args.device)
    cudnn.benchmark = True

    # data pipeline
    dataset_train = Kinetics(
        mode="pretrain",
        datafile_dir=args.datafile_dir,
        sampling_rate=args.sampling_rate,
        num_frames=args.num_frames,
        target_fps=args.target_fps,
        train_color_jitter=args.color_jitter,
        train_jitter_scales=(256, 320),
        train_crop_size=args.img_size,
        repeat_aug=args.repeat_aug,
        jitter_aspect_relative=args.jitter_aspect_relative,
        jitter_scales_relative=args.jitter_scales_relative,
    )

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    print(f"Sampler_train = {sampler_train}")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    # define the model
    model = models_mae.__dict__[args.model](**vars(args))
    model.to(device)
    model_without_ddp = model
    print(f"Model: {model_without_ddp}")
    print(f"Number of params (M): {(sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1.e6)}")

    # effective batch size
    eff_batch_size = args.batch_size_per_gpu * args.accum_iter * misc.get_world_size()
    print(f"Effective batch size: {eff_batch_size} = {args.batch_size_per_gpu} batch_size_per_gpu * {args.accum_iter} accum_iter * {misc.get_world_size()} GPUs")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay, bias_wd=args.bias_wd)
    optimizer = torch.optim._multi_tensor.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler(fp32=args.fp32)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        
        data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, args=args, fp32=args.fp32)

        if args.output_dir and (epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs):
            checkpoint_path = misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}

        if args.output_dir and misc.is_main_process():
            with pathmgr.open(f"{args.output_dir}/{args.save_prefix}_log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # prepare data files
    video_files = find_mp4_files(directories=args.data_dirs)
    
    if args.data_frac < 1.0:
        from math import ceil
        n_vids = len(video_files)
        n_vids_keep = ceil(n_vids * args.data_frac)
        video_files = video_files[:n_vids_keep]
        print(f"Training on {n_vids_keep} of {n_vids} video files.")
    else:
        n_vids = len(video_files)
        print(f"Training on all {n_vids} video files.")

    write_csv(video_files=video_files, save_dir=args.datafile_dir, save_name='train')

    # train
    main(args)