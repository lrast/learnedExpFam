import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn
from torch.nn import Parameter

from torch.nn.modules import batchnorm
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.func import vmap, jacrev, hessian


class ICNN(pl.LightningModule):
    """ Input convex neural network for basic learning experiments
    """
    def __init__(self, **kwargs):
        super(ICNN, self).__init__()

        hyperparameterValues = {
            # seed
            'seed': torch.random.seed(),
            # architecture
            'input_dim': 2,
            'hidden_dims': (10, 6),
            'dropout': 0.5,
            'CELU_alpha': 1.,
            'nonconvex': False,
            'batchnorm': False,
            'skipconnections': True,

            # training
            'lr': 1E-3,
            'batchsize': 32,

            # training data
            'num_samples': 2000,
            'variance': 1
        }

        hyperparameterValues.update(kwargs)
        self.save_hyperparameters(hyperparameterValues, ignore=['data_to_model'])

        if self.hparams.nonconvex:
            print("This network is not guaranteed to be convex")

        # print(self.hparams)

        torch.manual_seed(self.hparams.seed)
        np.random.seed(self.hparams.seed % (2**32-1))

        # model
        self.initialLayer = nn.Linear(self.hparams.input_dim, self.hparams.hidden_dims[0])

        internal_widths = tuple(self.hparams.hidden_dims) + (1,)
        self.internalLayers = nn.ModuleList([
                PositiveLinear(internal_widths[i], internal_widths[i+1])
                for i in range(len(internal_widths) - 1)
            ])

        if self.hparams.nonconvex:
            self.internalLayers = nn.ModuleList([
                    nn.Linear(internal_widths[i], internal_widths[i+1])
                    for i in range(len(internal_widths) - 1)
                ])

        if self.hparams.skipconnections:
            shortcut_outputs = tuple(self.hparams.hidden_dims[1:]) + (1,)
            self.shortcutLayers = nn.ModuleList([
                    nn.Linear(self.hparams.input_dim, output_size)
                    for output_size in shortcut_outputs
                ])

        if self.hparams.batchnorm:
            output_sizes = tuple(self.hparams.hidden_dims)
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(output_sizes[i])
                for i in range(len(internal_widths) - 1)
            ])

        self.nlin = nn.CELU(alpha=self.hparams.CELU_alpha)
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.lossFn = nn.MSELoss()

    def forward(self, y, scalar_outs=False):
        z = self.initialLayer(y)

        for i in range(len(self.internalLayers)):
            z = self.nlin(z)
            if self.hparams.batchnorm:
                z = self.batch_norms[i](z)

            z = self.dropout(z)
            z = self.internalLayers[i](z)
            if self.hparams.skipconnections:
                z = z + self.shortcutLayers[i](y)

        if scalar_outs:
            return z.sqeeze()
        return z

    def target(self, ts):
        """ compute value of the target function"""
        return (ts**2).sum(1)[:, None]

    # primal functions
    def fwd_cpu(self, xs):
        xs = torch.as_tensor(xs, dtype=torch.float32, device=self.device)
        return self.forward(xs).cpu()

    def jac(self, ts):
        """ Jacobian of the network """
        J = vmap(jacrev(self.fwd_cpu))
        return J(ts)[:, 0, :]

    def hess(self, ts):
        """Hessian of the network """
        H = vmap(hessian(self.fwd_cpu))
        return H(ts)[:, 0, :, :]

    # dual functions
    def dual_opt(self, p, optim_method=torch.optim.Adam):
        """
            Solve the dual optimization problem.

            Note that issues will arise if the desired slope is not achieved
            by the CGF
        """
        def to_minimize(x):
            return -(torch.einsum('Nk, Nk -> N', p, x) - self.fwd_cpu(x).squeeze())

        input_val = Parameter(torch.zeros(p.shape))
        optimizer = optim_method((input_val,), lr=1)
        
        for step in range(200):
            curr_val = optimizer.param_groups[0]['params'][0]
            optimizer.zero_grad()
            
            out = to_minimize(curr_val).sum()
            out.backward()
            optimizer.step()

        x_val = optimizer.param_groups[0]['params'][0].data
        
        return x_val, -to_minimize(x_val)

    def inv_jac(self, p, **kwargs):
        x, _ = self.dual_opt(p, **kwargs)
        return x

    def dual_function(self, p, **kwargs):
        _, values = self.dual_opt(p, **kwargs)
        return values

    # training 
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

    # Manage data
    def setup(self, stage=None):
        """ Generate samples of the empirical moment generating function
            for our dataset.
        """

        # first, we need to determine the t values that we want to sample.
        N_dims = 2

        self.ts = self.hparams.variance**0.5 * torch.randn((self.hparams.num_samples, N_dims))

        if N_dims == 1:  # edge case
            self.ts = self.ts[:, None]

        self.CGFs = self.target(self.ts)

        # set up the datasets
        full_dataset = TensorDataset(self.ts, self.CGFs)
        self.train_split, self.val_split = random_split(full_dataset, (0.8, 0.2))

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


class PositiveLinear(nn.Linear):
    """Positive linear layer for the internal pass of the ICNN"""
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__(in_features, out_features)

        self.nonLin = nn.Softplus()

    def forward(self, x):
        return nn.functional.linear(x, self.nonLin(self.weight), bias=self.bias)


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
