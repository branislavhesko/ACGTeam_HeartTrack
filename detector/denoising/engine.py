import os

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn as nn
import torch.optim as optim

from model import UNet
from config import DenoisingConfig
from dataloader import get_dataloader


class Engine(L.LightningModule):
    def __init__(self, model: nn.Module, config: DenoisingConfig, **kwargs):
        super().__init__(**kwargs)
        self.model = model.cuda()
        self.quality_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.63, 2.42]))

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
        signal, noise, targets = [x.cuda() for x in batch]
        self.model = self.model.to('cuda')
        quality_output = self.model(noise)

        loss = self.quality_criterion(quality_output, targets)

        prediction = torch.argmax(quality_output, dim=1)
        acc = (prediction == targets).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        torch.save({'model_state_dict': model.state_dict()}, f"checkpoints/model_{self.current_epoch}.pt")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    config = DenoisingConfig()
    model = UNet()
    engine = Engine(model, config)

    os.makedirs("checkpoints", exist_ok=True)

    trainer = L.Trainer(max_epochs=config.num_epochs)
    trainer.fit(engine)
