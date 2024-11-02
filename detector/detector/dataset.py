import dataclasses
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import pandas as pd


@dataclasses.dataclass
class Annotation:
    ppg_path: str
    annotation_path: str


class Dataset(torch.utils.data.Dataset):
    RATE = 30
    LENGTH = 304
    
    def __init__(self, annotations: List[Annotation]):
        self.annotations = annotations
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        ppg = np.loadtxt(annotation.ppg_path)
        ppg = ppg - np.mean(ppg)
        ppg = ppg / np.std(ppg)
        if len(ppg) < self.LENGTH:
            ppg_new = np.zeros(self.LENGTH) + np.mean(ppg)
            ppg_new[:len(ppg)] = ppg
            ppg = ppg_new
        with open(annotation.annotation_path, "r") as f:
            annotation = json.load(f)
        quality_labels = np.ones(len(ppg))
        click_labels = np.zeros(len(ppg))
        for point in annotation["points"]:
            click_labels[int(point["time"] * 30)] = 1
        for segment in annotation["bad_segments"]:
            if abs(segment["start"] - segment["end"]) > 0.5:
                start = max(0, int(segment["start"] * 30))
                end = min(int(segment["end"] * 30), len(quality_labels))
                quality_labels[start:end] = 0
                if quality_labels.sum() <= self.RATE:
                    quality_labels *= 0
        
        max_pool = torch.nn.MaxPool1d(5, 1, 2)
        click_labels = max_pool(torch.from_numpy(click_labels).unsqueeze(0)).squeeze(0)
        return torch.from_numpy(ppg).unsqueeze(0).float(), click_labels.long(), torch.from_numpy(quality_labels).long()


def load_annotations(path: str):
    csv_files = list(Path(path).glob("*.csv"))
    json_files = [str(p).replace(".csv", ".json") for p in csv_files]
    annotations = []
    for csv_file, json_file in zip(csv_files, json_files):
        if not Path(json_file).exists():
            raise FileNotFoundError(f"Annotation file {json_file} not found")
        annotations.append(Annotation(csv_file, json_file))
    print(f"Loaded {len(annotations)} annotations")
    return annotations


def get_dataloader(path: str, batch_size: int = 2, shuffle: bool = True):
    annotations = load_annotations(path)
    dataset = Dataset(annotations)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, persistent_workers=True)


if __name__ == "__main__":
    path = "/Users/brani/code/ACGTeam_HeartTrack_data/"
    from matplotlib import pyplot as plt
    dataloader = get_dataloader(path)
    for ppg, click_labels, quality_labels in dataloader:
        print(ppg.shape, click_labels.shape, quality_labels.shape)
        plt.subplot(3, 1, 1)
        plt.plot(ppg[0].squeeze(0))
        plt.subplot(3, 1, 2)
        plt.plot(click_labels[0].squeeze(0))
        plt.subplot(3, 1, 3)
        plt.plot(quality_labels[0].squeeze(0))
        plt.show()
        break
