# splitting out the working models that we will focus on.
import torch

from torch import nn
from torch.utils.data import DataLoader, random_split

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize

from models.baseline_adaptable import BaselineAdaptable

from functools import partial


class Basic_MNIST(BaselineAdaptable):
    """
        General Face angle estimation: classification
    """
    def __init__(self, **kwargs):
        super(Basic_MNIST, self).__init__(layer_ind=1)
        hyperparameterValues = {
            # architecture hyperparameters
            'pixelDim': 28,
            'hidden_dims': 28,
            # training hyperparameters
            'lr': 1E-3,
            'batchsize': 32,
            'max_epochs': 1000,
            'patience': 40,
            # stimulus generation hyperparameters
            'seed': torch.random.seed(),
        }
        hyperparameterValues.update(kwargs)
        self.save_hyperparameters(hyperparameterValues)

        # set seed
        torch.manual_seed(self.hparams.seed)

        self.model = nn.Sequential(
            nn.Linear(self.hparams.pixelDim**2, self.hparams.hidden_dims),
            nn.CELU(),
            nn.Linear(self.hparams.hidden_dims, 10)
        )

        self.lossFn = nn.CrossEntropyLoss()

    def forward(self, images):
        """ run the network """
        nsamples = images.shape[0]
        return self.model(images.view(nsamples, -1))

    def training_step(self, batch, batchidx):
        images, targets = batch

        prediction = self.forward(images)

        loss = self.lossFn(prediction, targets)
        self.log('Train Loss', loss.item())

        return loss

    def validation_step(self, batch, batchidx):
        images, targets = batch

        prediction = self.forward(images)

        loss = self.lossFn(prediction, targets)

        accuracy = (torch.argmax(prediction, axis=1) == targets)
        accuracy = accuracy.sum() / accuracy.shape[0]

        self.log_dict({'Val Loss': loss.item(),
                       'Val acc': accuracy})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # data
    def setup(self, stage=None):
        """generate the datasets"""
        try:  # skip reinitialize
            a = self.data_train
            return
        except AttributeError:
            print('setup called')
            pass

        mnist_data = MNIST('~/Datasets/', download=True, transform=Compose(
                          [ToTensor(), Normalize(0., 1.),
                           partial(torch.reshape, shape=(-1,))
                           ]))

        self.data_train, self.data_val = random_split(mnist_data, (0.9, 0.1))

    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.hparams.batchsize,
                          shuffle=True,
                          num_workers=8,
                          persistent_workers=True
                          )

    def val_dataloader(self):
        val_size = len(self.data_val)
        return DataLoader(self.data_val, batch_size=val_size,
                          num_workers=8,
                          persistent_workers=True
                          )
