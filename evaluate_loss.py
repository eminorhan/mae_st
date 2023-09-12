import os
import math
import random
import argparse
import torch
import json
import numpy as np
from torchvision.utils import save_image
import models_mae
from util.decoder.utils import tensor_normalize, spatial_sampling
import av

MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)

def get_args_parser():
    parser = argparse.ArgumentParser("Evaluate prediction loss on a directory of video clips", add_help=False)
    parser.add_argument("--mask_ratio", default=0.9, type=float, help="Masking ratio (percentage of removed patches)")
    parser.add_argument("--mask_type", default="temporal", type=str, help="mask type (temporal or random)", choices=["random", "temporal"])
    parser.add_argument("--clip_dir", default="", type=str, help="directory where eval video clips are stored")
    parser.add_argument("--output_dir", default="", type=str, help="output directory where output files are saved")
    parser.add_argument("--model_path", default="", type=str, help="path to pretrained model")
    parser.add_argument("--savefile_name", default="losses", type=str, help="name of file to save losses in")
    parser.add_argument("--model_arch", default="mae_vit_huge_patch14", type=str, help="Model architecture")
    parser.add_argument("--num_samples", default=1, type=int, help="number of samples to sample from a video clip")
    parser.add_argument("--plot", default=False, type=bool, help="plots images if true")
    parser.add_argument("--device", default="cuda", type=str, help="device to use for testing")

    return parser

def get_start_end_idx(video_size, clip_size, clip_idx, num_clips_uniform, use_offset=False):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips_uniform clips, and select the start and end index of clip_idx-th video clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips_uniform
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips_uniform == 1:
                # Take the center clip if num_clips_uniform is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(delta / (num_clips_uniform - 1))
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips_uniform
    end_idx = start_idx + clip_size - 1

    return start_idx, end_idx, start_idx / delta if delta != 0 else 0.0

def pyav_decode_stream(container, start_pts, end_pts, stream, stream_name, buffer_size=0):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts

def pyav_decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx,
    num_clips_uniform=10,
    target_fps=30,
    use_offset=False,
):
    """
    Convert the video from its original fps to the target_fps. If the video
    support selective decoding (contain decoding information in the video head),
    the perform temporal selective decoding and sample a clip from the video
    with the PyAV decoder. If the video does not support selective decoding,
    decode the entire video.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames.
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips_uniform
            clips, and select the clip_idx-th video clip.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): If True, the entire video was decoded.
    """
    # Try to fetch the decoding information from the video head. Some videos do not support fetching the decoding information, in that case it will get None duration.
    fps = float(container.streams.video[0].average_rate)
    frames_length = container.streams.video[0].frames
    duration = container.streams.video[0].duration

    if duration is None:
        # If failed to fetch the decoding information, decode the entire video.
        decode_all_video = True
        video_start_pts, video_end_pts = 0, math.inf
    else:
        # Perform selective decoding.
        decode_all_video = False
        clip_size = np.maximum(1.0, np.ceil(sampling_rate * (num_frames - 1) / target_fps * fps))
        start_idx, end_idx, fraction = get_start_end_idx(frames_length, clip_size, clip_idx, num_clips_uniform, use_offset=use_offset)
        timebase = duration / frames_length
        video_start_pts = int(start_idx * timebase)
        video_end_pts = int(end_idx * timebase)

    frames = None
    # If video stream was found, fetch video frames from the video.
    if container.streams.video:
        video_frames, max_pts = pyav_decode_stream(container, video_start_pts, video_end_pts, container.streams.video[0], {"video": 0})
        container.close()

        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.as_tensor(np.stack(frames))
    return frames, fps, decode_all_video

def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames

def prepare_video(path):
    video_container = av.open(path)
    frames, _, _ = pyav_decode(video_container, 8, 16, -1, num_clips_uniform=10, target_fps=25, use_offset=False)
    frames = temporal_sampling(frames, 0, 120, 16)
    frames = tensor_normalize(frames, torch.tensor(MEAN), torch.tensor(STD)).permute(3, 0, 1, 2)
    frames = spatial_sampling(
        frames,
        spatial_idx=-1,
        min_scale=224,
        max_scale=224,
        crop_size=224,
        random_horizontal_flip=False,
        inverse_uniform_sampling=False,
        aspect_ratio=[1,1],
        scale=[1,1],
        motion_shift=False,
    )
    return frames

def get_mp4_files(directory):
    mp4_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.mp4'):
                mp4_files.append(file)
    
    return mp4_files

if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    print(args)
    
    device = torch.device(args.device)

    # set up and load model
    model = models_mae.__dict__[args.model_arch](t_patch_size=2, cls_embed=True, norm_pix_loss=True, sep_pos_embed=True, decoder_depth=4)
    model.eval()

    checkpoint = torch.load(args.model_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model.to(device)  # move to device

    clips = get_mp4_files(args.clip_dir)
    losses = {}

    for clip in clips:
        # we're going to sample multiple clips from each video
        clip_losses = []

        for sample in range(args.num_samples):
            # load and prepate eval video
            vid = prepare_video(os.path.join(args.clip_dir, clip))
            clip_name = os.path.splitext(clip)[0]
            vid = vid.unsqueeze(0) 
            vid = vid.to(device)  # move to device
            
            with torch.no_grad():
                if args.plot:
                    loss, _, _, vis = model(vid, mask_ratio=args.mask_ratio, visualize=True, mask_type=args.mask_type)
                    print(f"MAE temporal prediction loss on {clip_name}: {loss.item()}")
                    clip_losses.append(loss.item())

                    vis = vis[0].permute(0, 2, 1, 3, 4)
                    a = vis[0, :, :, :, :]  # original
                    b = vis[1, :, :, :, :]  # masked
                    c = vis[2, :, :, :, :]  # reconstruction
                    vis = torch.cat((a, b, c), 0)
                    save_image(vis, f"{args.output_dir}/{clip_name}.jpg", nrow=16, padding=1, normalize=True, scale_each=True)
                else:
                    loss, _, _ = model(vid, mask_ratio=args.mask_ratio, visualize=False, mask_type=args.mask_type)
                    print(f"MAE temporal prediction loss on {clip_name}: {loss.item()}")
                    clip_losses.append(loss.item())

        losses[clip_name] = np.mean(clip_losses)

    with open(f"{args.output_dir}/{args.savefile_name}.json", "w") as json_file:
        json.dump(losses, json_file)