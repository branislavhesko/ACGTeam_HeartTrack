import dataclasses

import numpy as np
import torch
import pandas as pd


@dataclasses.dataclass
class Annotation:
    ppg_path: str
    annotation_path: str


class Dataset(torch.utils.data.Dataset):
    def __init__(self, annotations: List[Annotation]):
        self.annotations = annotations
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        ppg = np.load(annotation.ppg_path)
        annotation = pd.read_csv(annotation.annotation_path)
        return ppg, annotation
