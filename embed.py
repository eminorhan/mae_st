# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import os
import time
from pathlib import Path

import models_vit
import util.misc as misc
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from util.kinetics import Kinetics
from util.logging import master_print as print


def get_args_parser():
    parser = argparse.ArgumentParser("Embeddings from ViT models trained with MAE-ST", add_help=False)
    parser.add_argument("--batch_size_per_gpu", default=64, type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument("--save_prefix", default="", type=str, help="prefix for saving embeddings")

    # Model parameters
    parser.add_argument("--model", default="vit_large_patch16", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--drop_path_rate", type=float, default=0.1, metavar="PCT", help="Drop path rate (default: 0.1)")

    # Augmentation parameters
    parser.add_argument("--color_jitter", type=float, default=None, metavar="PCT", help="Color jitter factor (enabled only when not using Auto/RandAug)")
    parser.add_argument("--aa", type=str, default="rand-m7-mstd0.5-inc1", metavar="NAME", help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)")

    # * Random Erase params
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT", help="Random erase prob (default: 0.25)")
    parser.add_argument("--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")')
    parser.add_argument("--recount", type=int, default=1, help="Random erase count (default: 1)")
    parser.add_argument("--resplit", action="store_true", default=False, help="Do not random erase first (clean) augmentation split")

    # * Finetuning params
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument("--cls_token", action="store_false", dest="global_pool", help="Use class token instead of global pool for classification")
    parser.add_argument("--data_dirs", type=str, default=[""], nargs="+", help="Data paths")
    parser.add_argument("--datafile_dir", type=str, default="./datafiles", help="Store data files here")
    parser.add_argument("--output_dir", default="./embeddings", help="save embeddings here, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    # Video related configs
    parser.add_argument("--no_env", action="store_true")
    parser.add_argument("--rand_aug", default=False, action="store_true")
    parser.add_argument("--t_patch_size", default=2, type=int)
    parser.add_argument("--num_frames", default=32, type=int)
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--sampling_rate", default=2, type=int)
    parser.add_argument("--repeat_aug", default=1, type=int)
    parser.add_argument("--cpu_mix", action="store_true")
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--bias_wd", action="store_true")
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument("--fp32", action="store_true")
    parser.set_defaults(fp32=True)
    parser.add_argument("--jitter_scales_relative", default=[0.9, 1.0], type=float, nargs="+")
    parser.add_argument("--jitter_aspect_relative", default=[1.0, 1.8], type=float, nargs="+")
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)

    return parser

def find_mp4_files(directories):
    """Recursively search for .mp4 files in directories and their subdirectories"""
    mp4_files = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".mp4"):
                    mp4_files.append((os.path.join(root, file), os.path.basename(root)))
    return mp4_files

def find_webm_files(directories):
    """Recursively search for .mp4 files in directories and their subdirectories"""
    webm_files = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".webm"):
                    webm_files.append((os.path.join(root, file), os.path.basename(root)))
    return webm_files

def write_csv(video_files, save_dir, save_name):
    """Write the .csv file with video path and subfolder index"""
    with open(os.path.join(save_dir, f'{save_name}.csv'), 'w', newline='') as csvfile:
        for video_file, _ in video_files:
            csvfile.write(f"{video_file}, {os.path.splitext(os.path.basename(video_file))[0]}\n")

def embed(data_loader, model, device, fp32=True):

    embeddings = []
    labels = []

    # switch to evaluation mode
    model.eval()

    for it, (images, target) in enumerate(data_loader):

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if len(images.shape) == 6:
            b, r, c, t, h, w = images.shape
            images = images.view(b * r, c, t, h, w)
            target = target.view(b * r)

        # compute output
        with torch.cuda.amp.autocast(enabled=not fp32):
            output = model(images)

        embeddings.append(output)
        labels.append(target)

        if it % 99 == 0: print('Iter:', it)

        if it == 299: break

    embeddings = torch.cat(embeddings, 0)
    embeddings = embeddings.cpu().numpy()

    labels = torch.cat(labels, 0)
    labels = labels.cpu().numpy()

    print('Embeddings shape, min, max:', embeddings.shape, embeddings.min(), embeddings.max())
    print('Labels shape, min, max:', labels.shape, labels.min(), labels.max())

    return embeddings, labels

def main(args):
    misc.init_distributed_mode(args)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_val = Kinetics(
        mode="val",
        datafile_dir=args.datafile_dir,
        sampling_rate=args.sampling_rate,
        num_frames=args.num_frames,
        train_jitter_scales=(256, 320),
        jitter_aspect_relative=args.jitter_aspect_relative,
        jitter_scales_relative=args.jitter_scales_relative,
    )

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if len(dataset_val) % num_tasks != 0:
        print("Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. This will slightly alter validation results as extra duplicate entries are added to achieve equal num of samples per-process.")
    
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    model = models_vit.__dict__[args.model](**vars(args))
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print("Number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = (args.batch_size_per_gpu * misc.get_world_size())
    print("effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    model_without_ddp = model.module

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=None, loss_scaler=None)

    start_time = time.time()
    with torch.no_grad():
        embeddings, labels = embed(data_loader_val, model, device)
    np.savez(os.path.join(args.output_dir, args.save_prefix + "_embeddings.npz"), embeddings=embeddings, labels=labels)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Total time {}".format(total_time_str))
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.eval = True

    # prepare data files
    val_files = find_webm_files(directories=args.data_dirs)
    write_csv(video_files=val_files, save_dir=args.datafile_dir, save_name='val')

    # run
    main(args)