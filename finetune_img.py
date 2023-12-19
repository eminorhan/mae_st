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
import json
import os
import time
from pathlib import Path

import models_vit_img
import util.misc as misc
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DistributedSampler, DataLoader
from iopath.common.file_io import g_pathmgr as pathmgr
from engine_finetune_img import evaluate, train_one_epoch
import util.lr_decay as lrd

from util.decoder.mixup import MixUp as MixVideo
from util.kinetics import Kinetics
from util.logging import master_print as print
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import trunc_normal_

def get_args_parser():
    parser = argparse.ArgumentParser("MAE fine-tuning for image classification", add_help=False)
    parser.add_argument("--batch_size_per_gpu", default=64, type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--accum_iter", default=1, type=int, help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument("--save_prefix", default="", type=str, help="prefix for saving checkpoint and log files")

    # Model parameters
    parser.add_argument("--model", default="vit_large_patch16", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--drop_path_rate", type=float, default=0.1, metavar="PCT", help="Drop path rate")

    # Optimizer parameters
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM", help="Clip gradient norm (default: None, no clipping)")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR", help="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    parser.add_argument("--layer_decay", type=float, default=0.8, help="layer-wise lr decay from ELECTRA/BEiT")
    parser.add_argument("--min_lr", type=float, default=1e-5, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0")
    parser.add_argument("--warmup_epochs", type=int, default=0, metavar="N", help="epochs to warmup LR")

    # Augmentation parameters
    parser.add_argument("--color_jitter", type=float, default=None, metavar="PCT", help="Color jitter factor (enabled only when not using Auto/RandAug)")
    parser.add_argument("--aa", type=str, default="rand-m7-mstd0.5-inc1", metavar="NAME", help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)")

    # * Random Erase params
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT", help="Random erase prob (default: 0.25)")
    parser.add_argument("--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")')
    parser.add_argument("--recount", type=int, default=1, help="Random erase count (default: 1)")
    parser.add_argument("--resplit", action="store_true", default=False, help="Do not random erase first (clean) augmentation split")

    # * Mixup params
    parser.add_argument("--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0.")
    parser.add_argument("--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0.")
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None, help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)")
    parser.add_argument("--mixup_prob", type=float, default=1.0, help="Probability of performing mixup or cutmix when either/both is enabled")
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5, help="Probability of switching to cutmix when both mixup and cutmix enabled")
    parser.add_argument("--mixup_mode", type=str, default="batch", help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=False)
    parser.add_argument("--cls_token", action="store_false", dest="global_pool", help="Use class token instead of global pool for classification")
    parser.add_argument("--num_classes", default=700, type=int, help="number of the classes")
    parser.add_argument("--train_dir", default="", help="path to train data")
    parser.add_argument("--val_dir", default="", help="path to val data")
    parser.add_argument("--datafile_dir", type=str, default="./datafiles", help="Store data files here")
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
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
    parser.add_argument("--jitter_scales_relative", default=[0.08, 1.0], type=float, nargs="+")
    parser.add_argument("--jitter_aspect_relative", default=[0.75, 1.3333], type=float, nargs="+")
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)

    return parser

def list_subdirectories(directory):
    subdirectories = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            subdirectories.append(entry.path)
    subdirectories.sort()  # Sort the list of subdirectories alphabetically
    return subdirectories

def find_mp4_files(directory):
    """Recursively search for .mp4 or .webm files in a directory"""
    mp4_files = []
    subdir_idx = 0
    subdirectories = list_subdirectories(directory)
    for subdir in subdirectories:
        files = os.listdir(subdir)
        files.sort()
        for file in files:
            if file.endswith((".mp4", ".webm")):
                mp4_files.append((os.path.join(subdir, file), subdir_idx))
        subdir_idx += 1
    return mp4_files

def write_csv(video_files, save_dir, save_name):
    """Write the .csv file with video path and subfolder index"""
    with open(os.path.join(save_dir, f'{save_name}.csv'), 'w', newline='') as csvfile:
        for video_file, subdir_idx in video_files:
            csvfile.write(f"{video_file}, {subdir_idx}\n")

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

    dataset_train = Kinetics(
        mode="finetune",
        datafile_dir=args.datafile_dir,
        sampling_rate=args.sampling_rate,
        num_frames=args.num_frames,
        train_jitter_scales=(256, 320),
        repeat_aug=args.repeat_aug,
        pretrain_rand_erase_prob=args.reprob,
        pretrain_rand_erase_mode=args.remode,
        pretrain_rand_erase_count=args.recount,
        pretrain_rand_erase_split=args.resplit,
        aa_type=args.aa,
        rand_aug=args.rand_aug,
        jitter_aspect_relative=args.jitter_aspect_relative,
        jitter_scales_relative=args.jitter_scales_relative,
    )
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
    sampler_train = DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))

    if len(dataset_val) % num_tasks != 0:
        print("Warning: Enabling distributed evaluation with an eval dataset not divisible by process number."
              "This will slightly alter validation results as extra duplicate entries are added to achieve equal num of samples per-process.")
        
    sampler_val = DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    data_loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=23*args.batch_size_per_gpu, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = MixVideo(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, mix_prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, label_smoothing=args.smoothing, num_classes=args.num_classes)

    model = models_vit_img.__dict__[args.model](num_classes=args.num_classes, global_pool=args.global_pool)

    if misc.get_last_checkpoint(args) is None and args.finetune and not args.eval:
        with pathmgr.open(args.finetune, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if "model" in checkpoint.keys():
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint["model_state"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = (args.batch_size_per_gpu * args.accum_iter * misc.get_world_size() * args.repeat_aug)

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay, no_weight_decay_list=model_without_ddp.no_weight_decay(), layer_decay=args.layer_decay)    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler(fp32=args.fp32)

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):

        data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args.clip_grad, mixup_fn, args=args, fp32=args.fp32)
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        if args.output_dir and test_stats["acc1"] > max_accuracy:
            print('Improvement in max test accuracy. Saving model!')
            checkpoint_path = misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, **{f"test_{k}": v for k, v in test_stats.items()}, "epoch": epoch, "n_parameters": n_parameters}

        if args.output_dir and misc.is_main_process():
            with pathmgr.open(f"{args.output_dir}/{args.save_prefix}_log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    return [checkpoint_path]

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # prepare data files
    train_files = find_mp4_files(directory=args.train_dir)
    write_csv(video_files=train_files, save_dir=args.datafile_dir, save_name='train')
    val_files = find_mp4_files(directory=args.val_dir)
    write_csv(video_files=val_files, save_dir=args.datafile_dir, save_name='val')

    # finetune
    main(args)