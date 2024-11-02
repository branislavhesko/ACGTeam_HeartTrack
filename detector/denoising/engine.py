import pytorch_lightning as L
import torch

import torch.nn as nn
import torch.optim as optim

from model import UNet
from config import DenoisingConfig
from dataloader import get_dataloader


class Engine(L.LightningModule):
    def __init__(self, model: nn.Module, config: DenoisingConfig, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.denoising_criterion = nn.MSELoss()
        self.quality_criterion = nn.BCEWithLogitsLoss()

        self.data_loaders = get_dataloader(
            folder_path=config.folder_path,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True
        )

        self.lr = config.lr

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.data_loaders

    def training_step(self, batch):
        data, targets = batch
        denoised_output, quality_output = self.model(data)

        denoising_loss = self.denoising_criterion(denoised_output, data)
        quality_loss = self.quality_criterion(quality_output, targets.float().unsqueeze(1))
        loss = denoising_loss + quality_loss

        preds = (quality_output > 0.5).float()
        acc = (preds == targets).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_denoising_loss', denoising_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_quality_loss', quality_loss, on_step=True, on_epoch=True, prog_bar=True)

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
