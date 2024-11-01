import pytorch_lightning as L
import torch

import torch.nn as nn
import torch.optim as optim

from model import UNet
from config import DenoisingConfig
from dataloader import get_dataloader


class Engine(L.LightningModule):
    def __init__(self, model, config, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.criterion = nn.MSELoss()

        self.data_loaders = get_dataloader(
            folder_path=config.folder_path,
            batch_size=config.batch_size,
            num_workers=1,
            pin_memory=True
        )

        self.lr = config.lr

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.data_loaders

    def training_step(self, batch):
        data, targets = batch
        denoised_output, quality_output = self.model(data)

        loss = self.criterion(denoised_output, data)
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    config = DenoisingConfig()
    model = UNet()
    engine = Engine(model, config)

    trainer = L.Trainer(max_epochs=config.num_epochs)
    trainer.fit(engine)
