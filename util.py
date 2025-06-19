from decord import VideoReader
from decord import cpu
import numpy as np

import torchvision.transforms as transforms
from transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)


def loadvideo_decord(sample, sample_rate_scale=1,new_width=384, new_height=384, clip_len=8, frame_sample_rate=2,num_segment=1):
    fname = sample
    vr = VideoReader(fname, width=new_width, height=new_height,
                                     num_threads=1, ctx=cpu(0))
    converted_len = int(clip_len * frame_sample_rate)
    seg_len = len(vr) //num_segment
    duration = max(len(vr) // vr.get_avg_fps(),8)

    all_index = []
    for i in range(num_segment):
        index = np.linspace(0, seg_len, num=int(duration))
        index = np.clip(index, 0, seg_len - 1).astype(np.int64)
        index = index + i*seg_len
        all_index.extend(list(index))

    all_index = all_index[::int(sample_rate_scale)]
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer

def loadvideo_decord_origin(sample, sample_rate_scale=1,new_width=384, new_height=384, clip_len=8, frame_sample_rate=2,num_segment=1):
    fname = sample
    vr = VideoReader(fname, 
                                     num_threads=1, ctx=cpu(0))
    converted_len = int(clip_len * frame_sample_rate)
    seg_len = len(vr) //num_segment
    duration = max(len(vr) // vr.get_avg_fps(),8)

    all_index = []
    for i in range(num_segment):
        index = np.linspace(0, seg_len, num=int(duration))
        index = np.clip(index, 0, seg_len - 1).astype(np.int64)
        index = index + i*seg_len
        all_index.extend(list(index))

    all_index = all_index[::int(sample_rate_scale)]
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer


def load_video_segment(fname, start_sec, duration_sec, fps=1):

    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    original_fps = vr.get_avg_fps()
    
    start_frame = int(start_sec * original_fps)
    end_frame = int((start_sec + duration_sec) * original_fps)
    
    end_frame = min(end_frame, len(vr) - 1)
    
    num_frames_to_sample = int(duration_sec * fps)
    frame_indices = np.linspace(start_frame, end_frame, num_frames_to_sample, dtype=int)
    
    frame_indices = np.unique(frame_indices)
    frame_indices = np.clip(frame_indices, 0, len(vr) - 1)
    
    buffer = vr.get_batch(frame_indices).asnumpy()
    return buffer
