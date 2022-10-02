import torch
import os
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import JPEGDatasetTrain

from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import MeanAbsoluteError


BATCH_SIZE = 32


class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.ReLU(True),
        )

        self.val_accuracy = MeanAbsoluteError()

    def forward(self, x):
        return self.net(x).view(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch

        outs = self(x)
        loss = F.l1_loss(outs, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        outs = self(x)
        loss = F.l1_loss(outs, y)

        self.val_accuracy.update(outs, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        train_ds = JPEGDatasetTrain('../DIV2K_valid_HR')
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=os.cpu_count())

        return train_loader

    def val_dataloader(self):
        val_ds = JPEGDatasetTrain('../DIV2K_valid_HR')
        return DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=os.cpu_count())


def main():

    mnist_model = MNISTModel()

        # Initialize a trainer
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=3,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )

    # Train the model ⚡
    trainer.fit(mnist_model)



if __name__ == '__main__':
    main()
