import dataclasses

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet
from torchvision.transforms import Compose, ColorJitter,ToTensor, Resize


@dataclasses.dataclass
class FingerPresenceTrainerConfig:
    folder_path: str
    batch_size: int
    num_workers: int
    pin_memory: bool


def get_loader(
    folder_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool
):
    dataset = ImageFolder(folder_path)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True
    )
    

def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class FingerPresenceTrainer:
    
    def __init__(self, config: FingerPresenceTrainerConfig) -> None:
        self.model = efficientnet.efficientnet_b0(weights="DEFAULT")
        self.model.classifier[1] = nn.Linear(1280, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loader = get_loader(config.folder_path, config.batch_size, config.num_workers, config.pin_memory)
        self.transform = Compose([
            ToTensor(),
            Resize((224, 224))
        ])
    
    
    def train(self):
        self.model.train()
        for record, quality in self.loader:
            record = record.to(device())
            quality = quality.to(device())
            self.optimizer.zero_grad()
            output = self.model(record)
            loss = self.criterion(output, quality)
            
    def export(self, path: str):
        torch.onnx.export(self.model, torch.randn(1, 3, 224, 224).to(device()), path)


if __name__ == "__main__":
    trainer = FingerPresenceTrainer(FingerPresenceTrainerConfig(
        folder_path="/Users/brani/Downloads/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0",
        batch_size=1,
        num_workers=1,
        pin_memory=True
    ))
    trainer.train()
    trainer.export("finger_presence_classifier.onnx")