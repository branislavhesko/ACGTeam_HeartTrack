import dataclasses
from enum import Enum
from typing import Tuple

from torch.utils.data import DataLoader

from dataset import PPGDataset


class Mode(Enum):
    TRAIN = "train"
    VAL = "val"


@dataclasses.dataclass
class DenoisingConfig:
    lr: float = 1e-4
    batch_size: int = ...
    num_epochs: int = 15
    num_workers: int = 4

    def make_dataloaders(self):
        return {
            Mode.TRAIN: DataLoader(PPGDataset(...),
                                   batch_size=self.batch_size,
                                   num_workers=self.num_workers,
                                   shuffle=True
                                   )
        }

