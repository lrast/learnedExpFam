import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader, random_split

from scipy.stats import multivariate_normal


class CGF_ICNN(pl.LightningModule):
    """ Input convex neural network implementation for learning 
        cumulant generating functions
    """
    def __init__(self, data_to_model, **kwargs):
        super(CGF_ICNN, self).__init__()

        self.data = torch.tensor(data_to_model, dtype=torch.float32)

        hyperparameterValues = {
            'inputDim': 2,
            'hiddenDims': (10, 6),
            'lr': 1E-3,
            'batchsize': 32,
            'patience': 50
        }
        hyperparameterValues.update(kwargs)
        self.save_hyperparameters(hyperparameterValues, ignore=['data_to_model'])

        self.initialLayer = nn.Linear(self.hparams.inputDim, self.hparams.hiddenDims[0])

        self.internalLayers = nn.ModuleList([
                PositiveLinear(self.hparams.hiddenDims[0], self.hparams.hiddenDims[1]),
                PositiveLinear(self.hparams.hiddenDims[1], 1)
            ])

        self.shortcutLayers = nn.ModuleList([
                nn.Linear(self.hparams.inputDim, self.hparams.hiddenDims[1]),
                nn.Linear(self.hparams.inputDim, 1)
            ])

        self.nlin = nn.CELU()
        self.lossFn = nn.MSELoss()

    def forward(self, y):
        z = self.initialLayer(y)

        for i in range(len(self.internalLayers)):
            z = self.internalLayers[i](z) + self.shortcutLayers[i](y)
            z = self.nlin(z)

        return z

    def training_step(self, batch, batchidx):
        xs, ys = batch

        loss = self.lossFn(self.forward(xs), ys)
        self.log('Train Loss', loss)
        return loss

    def validation_step(self, batch, batchidx):
        xs, ys = batch
        loss = self.lossFn(self.forward(xs), ys)
        self.log('Val Loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def setup(self, stage=None):
        """ Generate samples of the empirical moment generating function
            for our dataset.
        """

        def make_CGF(ts, data):
            """ compute empirical CGF """
            outs = torch.exp(data @ ts.T).mean(axis=0)
            return torch.log(outs)

        # first, we need to determine the t values that we want to sample. 
        N_dims = self.data.shape[1]

        def search_width(n_axis):
            """ Find the range of input values for the moment generating function """
            fraction = 0.95

            def axis_samples(samples):
                all_data = []
                for ind in range(N_dims):
                    if ind == n_axis:
                        all_data.append(samples)
                    else:
                        all_data.append(torch.zeros(len(samples)))
                return torch.stack(all_data).T

            data_min = self.data[:, n_axis].min()
            data_max = self.data[:, n_axis].max()

            width = 1
            while width < 10:
                ax_vals = torch.linspace(-width, width, 50+10*width)
                ts = axis_samples(ax_vals)
                CGF_vals = make_CGF(ts, self.data)

                min_slope = (CGF_vals[1] - CGF_vals[0]) / (ax_vals[1] - ax_vals[0])
                max_slope = (CGF_vals[-1] - CGF_vals[-2]) / (ax_vals[-1] - ax_vals[-2])

                #print(width, min_slope, max_slope)

                stop = True
                if min_slope / data_min < fraction:
                    width += 1
                    stop = False
                elif max_slope / data_max < fraction:
                    width += 1
                    stop = False
                if stop:
                    break

            return width

        dim_variances = []
        for dim in range(N_dims):
            width = search_width(dim)
            dim_variances.append((width/1.5)**2)  # width is two standard deviations

        self.ts = torch.tensor(
                                multivariate_normal(np.zeros(N_dims), np.diag(dim_variances)
                                                    ).rvs(1000*N_dims),
                                dtype=torch.float32
                             )
        if N_dims == 1:  # edge case
            self.ts = self.ts[:, None]

        self.CGFs = make_CGF(self.ts, self.data)[:, None]

        # set up the datasets
        full_dataset = TensorDataset(self.ts, self.CGFs)
        l = len(full_dataset)
        self.train_split, self.val_split = random_split(
            full_dataset, (8*l//10, 2*l//10))

    def train_dataloader(self):
        return DataLoader(self.train_split,
                          batch_size=self.hparams.batchsize,
                          shuffle=True,
                          num_workers=2,
                          persistent_workers=True
                          )

    def val_dataloader(self):
        val_size = len(self.val_split)
        return DataLoader(self.val_split, batch_size=val_size,
                          num_workers=2,
                          persistent_workers=True
                          )


class PositiveLinear(nn.Module):
    """Positive linear layer for the internal pass of the ICNN"""
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.nonLin = nn.Softplus()

        self.weight = Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        """ Some fiddling with the initialization might be required
            because these are positive matrices that are applied repeatedly
        """
        self.weight.data.normal_(0., 1./(self.in_features))

    def forward(self, x):
        return nn.functional.linear(x, self.nonLin(self.weight))


class ICN_layer(nn.Module):
    """ individual layer of an ICNN.
        passes forward the input values 
    """
    def __init__(self, input_size, pre_size, post_size):
        super(ICN_layer, self).__init__()
        self.internal = nn.Linear(pre_size, post_size)
        self.shortcut = nn.Linear(input_size, post_size)

    def forward(self, x, original):
        return self.internal(x) + self.shortcut(original), original

