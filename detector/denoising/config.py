import dataclasses
from enum import Enum
from typing import Tuple

from torch.utils.data import DataLoader

# from dataset import PPGDataset


class Mode(Enum):
    TRAIN = "train"
    VAL = "val"


@dataclasses.dataclass
class DenoisingConfig:
    lr: float = 7e-5
    batch_size: int = 16
    num_epochs: int = 40
    num_workers: int = 1

    folder_path: str = "C:/Users/vojta/Downloads/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0"
    # folder_path: str = "C:/Users/vojta/Downloads/brno-university-of-technology-smartphone-ppg-database-but-ppg-1.0.0"