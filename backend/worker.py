import dataclasses
from functools import partial
import multiprocessing as mp
from typing import Callable

import cv2
import numpy as np
import torch

class VideoQueueItem:
    device_id: str
    video_path: str
    
    
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(torch.from_numpy(frame / 255.0).permute(2, 0, 1).float())
    cap.release()
    return torch.stack(frames)


def extract_fn(frames, roi_size: int = None, roi_center=None):
    print(frames.shape)
    if roi_size is not None:
        center_x = frames.shape[3] // 2 if roi_center is None else roi_center[0]
        center_y = frames.shape[2] // 2 if roi_center is None else roi_center[1]
        crops = frames[:, :, center_y - roi_size // 2:center_y + roi_size // 2, center_x - roi_size // 2:center_x + roi_size // 2]
    else:
        crops = frames
    print(crops.shape)
    return crops[:, 2, ...].mean(dim=(1, 2))


def process_video(video_path):
    video = load_video(video_path)
    signal = extract_fn(video, roi_size=50)
    print(signal.shape)
    with open(str(video_path).replace(".mp4", ".csv"), "w") as f:
        for s in signal.tolist():
            f.write(f"{s}\n")
    return signal


class Worker(mp.Process):
    def __init__(self, processing_fn, queue: mp.Queue):
        super().__init__()
        self.processing_fn = processing_fn
        self.input_queue = mp.Queue(maxsize=4)
        self.output_queue = mp.Queue(maxsize=4)
        self.stop_event = mp.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                item = self.input_queue.get()
                output = self.processing_fn(item)
                self.output_queue.put(output)
            except Exception as e:
                print(e)

    def stop(self):
        self.stop_event.set()

    def put(self, item):
        self.input_queue.put(item)

    def get(self):
        return self.output_queue.get()
