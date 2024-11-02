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
    lr: float = 1e-4
    batch_size: int = 2
    num_epochs: int = 30
    num_workers: int = 1

    folder_path: str = "C:/Users/vojta/Downloads/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0"

    # def make_dataloaders(self):
    #     return {
    #         Mode.TRAIN: DataLoader(PPGDataset(...),
    #                                batch_size=self.batch_size,
    #                                num_workers=self.num_workers,
    #                                shuffle=True
    #                                )
    #     }
