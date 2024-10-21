# splitting out the working models that we will focus on.

import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn
from torch.utils.data import DataLoader

from models.data_rotated_olivetti import FaceDataset


class FaceAngle(pl.LightningModule):
    """
        General Face angle estimation: classification

    """
    def __init__(self, N_classes=4, **kwargs):
        super(FaceAngle, self).__init__()
        hyperparameterValues = {
            # architecture hyperparameters
            'hidden_dims': (60, 20),
            # training hyperparameters
            'lr': 1E-3,
            'batchsize': 32,
            'max_epochs': 1500,
            'gradient_clip_val': 0.5,
            'patience': 200,
            # stimulus generation hyperparameters
            'N_classes': N_classes,
            'train_val_test_split': (300, 50, 50),
            'pixelDim': 64,
            'seed': torch.random.seed(),
        }
        hyperparameterValues.update(kwargs)
        self.save_hyperparameters(hyperparameterValues)

        # set seeds
        torch.manual_seed(self.hparams.seed)
        np.random.seed(self.hparams.seed % (2**32-1))

        self.model = nn.Sequential(
            nn.Linear(self.hparams.pixelDim**2, self.hparams.hidden_dims[0]),
            nn.CELU(),
            nn.Linear(self.hparams.hidden_dims[0], self.hparams.hidden_dims[1]),
            nn.CELU(),
            nn.Linear(self.hparams.hidden_dims[1], 4)
        )

        self.lossFn = nn.CrossEntropyLoss()

    def forward(self, images):
        """ run the network """
        nsamples = images.shape[0]
        return self.model(images.view(nsamples, -1))

    def training_step(self, batch, batchidx):
        images = batch['image']
        targets = batch['angle']

        prediction = self.forward(images)

        loss = self.lossFn(prediction, targets)
        self.log('Train Loss', loss.item())

        return loss

    def validation_step(self, batch, batchidx):
        images = batch['image']
        targets = batch['angle']

        prediction = self.forward(images)

        loss = self.lossFn(prediction, targets)

        accuracy = (torch.argmax(prediction, axis=1) == targets)
        accuracy = accuracy.sum() / accuracy.shape[0]

        self.log_dict({'Val Loss': loss.item(),
                       'Val acc': accuracy})

    def test_step(self, batch, batchidx):
        images = batch['image']
        targets = batch['angle']

        prediction = self.forward(images)

        loss = self.lossFn(prediction, targets)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # data
    def setup(self, stage=None):
        """generate the datasets"""
        numbers = self.hparams.train_val_test_split

        # generate datasets
        self.trainData = FaceDataset(self.hparams.N_classes, split='train',
                                     numbers=numbers)
        self.valData = FaceDataset(self.hparams.N_classes, split='validation',
                                   numbers=numbers)
        self.testData = FaceDataset(self.hparams.N_classes, split='test',
                                    numbers=numbers)

    def train_dataloader(self):
        return DataLoader(self.trainData,
                          batch_size=self.hparams.batchsize,
                          shuffle=True,
                          num_workers=8,
                          persistent_workers=True
                          )

    def val_dataloader(self):
        val_size = self.valData.angles.shape[0]
        return DataLoader(self.valData, batch_size=val_size,
                          num_workers=2,
                          persistent_workers=True
                          )

    def test_dataloader(self):
        test_size = self.testData.angles.shape[0]
        return DataLoader(self.testData, batch_size=test_size)

    def teardown(self, stage):
        """ clearing the memory after it is used """
        if stage == "fit":
            del self.trainData
            del self.valData
            del self.testData
