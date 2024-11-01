import pytorch_lightning as L
import torch

import torch.nn as nn
import torch.optim as optim

from model import UNet1D
from config import DenoisingConfig, Mode


class Engine(L.LightningModule):
    def __init__(self, model, config):
        super(Engine, self).__init__()
        self.model = model
        self.criterion = nn.MSELoss()

        self.data_loaders = self.args.make_dataloaders()

        self.lr = config.lr

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.data_loaders[Mode.TRAIN]

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.data_loaders[Mode.VAL]
    
    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self(data)
        loss = self.criterion(outputs, targets)

        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self(data)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    config = DenoisingConfig()
    model = UNet1D()
    engine = Engine(model, config)
    
    trainer = L.Trainer(max_epochs=50, accumulate_grad_batches=8, precision="16-mixed")
    trainer.fit(engine)
    