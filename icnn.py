import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn
from torch.nn import Parameter

from torch.nn.functional import leaky_relu
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.func import vmap, jacrev, hessian

from convex_initialization import ConvexLinear

from scipy.stats import multivariate_normal


class CGF_ICNN(pl.LightningModule):
    """ Input convex neural network implementation for learning 
        cumulant generating functions
    """
    def __init__(self, data_to_model, **kwargs):
        super(CGF_ICNN, self).__init__()

        self.data = data_to_model.clone().detach()

        hyperparameterValues = {
            # seed
            'seed': torch.random.seed(),
            # architecture
            'inputDim': 2,
            'hidden_dims': (200, 200, 200, 10),

            # training
            'lr': 1E-3,
            'batchsize': 128,
            'patience': 100,
            'bias_noise': 0.5,

            # training data
            'numsamples': 2000,
            'variance': None,
            'testmode': False,

            # to implement
            'skipconnections': False,
            'dropout': 0.0
        }
        hyperparameterValues.update(kwargs)
        self.save_hyperparameters(hyperparameterValues, ignore=['data_to_model'])

        torch.manual_seed(self.hparams.seed)
        np.random.seed(self.hparams.seed % (2**32-1))

        if self.hparams.skipconnections:
            raise NotImplementedError('Skip connections are not implemented')
        if self.hparams.dropout:
            raise NotImplementedError('Dropout is not implemented')

        if self.hparams.testmode:
            self.model = nn.Sequential(
                    nn.Linear(self.hparams.inputDim, 200),
                    LeakySoftplus(),
                    ConvexLinear(200, 200, bias_noise=self.hparams.bias_noise),
                    LeakySoftplus(),
                    ConvexLinear(200, 200, bias_noise=self.hparams.bias_noise),
                    LeakySoftplus(),
                    ConvexLinear(200, 1, bias_noise=self.hparams.bias_noise)
                )

        else:
            self.initial_layer = nn.Linear(self.hparams.inputDim,
                                           self.hparams.hidden_dims[0])

            in_sizes = self.hparams.hidden_dims
            out_sizes = self.hparams.hidden_dims[1:] + (1,)

            self.positive_layers = nn.ModuleList([
                    ConvexLinear(in_sizes[i], out_sizes[i],
                                 convex_init=True,
                                 bias_noise=self.hparams.bias_noise,
                                 )
                    for i in range(len(in_sizes))
                ])

            self.nlin = nn.LeakySoftplus()

        self.lossFn = nn.MSELoss()

    def forward(self, x, scalar_outs=False):
        if self.hparams.testmode:
            return self.model(x)

        else:
            z = self.initial_layer(x)

            for positive_layer in self.positive_layers:
                z = self.nlin(z)
                z = positive_layer(z)

            if scalar_outs:
                return z.sqeeze()
            return z

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
    def empirical_CGF(self, ts):
        """ compute empirical CGF """
        num_points = torch.tensor(self.data.shape[0])

        outs = torch.logsumexp(self.data @ ts.T, 0) - torch.log(num_points)
        return outs

    def setup(self, stage=None):
        """ Generate samples of the empirical moment generating function
            for our dataset.
        """

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
            while width < 10:  # is 10 actually a reasonable maximum value?
                ax_vals = torch.linspace(-width, width, 50+10*width)
                ts = axis_samples(ax_vals)
                CGF_vals = self.empirical_CGF(ts)

                min_slope = (CGF_vals[1] - CGF_vals[0]) / (ax_vals[1] - ax_vals[0])
                max_slope = (CGF_vals[-1] - CGF_vals[-2]) / (ax_vals[-1] - ax_vals[-2])

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

        if self.hparams.variance is None:
            dim_variances = []
            for dim in range(N_dims):
                width = search_width(dim)
                dim_variances.append((width/1.5)**2)  # width is two standard deviations
        else:
            dim_variances = [self.hparams.variance, self.hparams.variance]

        self.ts = torch.tensor(
                                multivariate_normal(np.zeros(N_dims), np.diag(dim_variances)
                                                    ).rvs(self.hparams.numsamples),
                                dtype=torch.float32
                             )

        if N_dims == 1:  # edge case
            self.ts = self.ts[:, None]

        self.CGFs = self.empirical_CGF(self.ts)[:, None]

        # set up the datasets
        full_dataset = TensorDataset(self.ts, self.CGFs)
        l = len(full_dataset)
        self.train_split, self.val_split = random_split(
            full_dataset, (0.8, 0.2))

    # alternative training data.
    def target(self, ts):
        """ compute value of the target function"""
        return (ts**2).sum(1)[:, None]

    def setup_simple(self, stage=None):
        """ Generate samples of the empirical moment generating function
            for our dataset.
        """

        # first, we need to determine the t values that we want to sample.
        N_dims = 2

        self.ts = 1.0**0.5 * torch.randn((self.hparams.numsamples, N_dims))

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


class LeakySoftplus(nn.Module):
    """ Smooth softplus version of the leaky ReLU
        negative_slope: slope of the negative component
        offset: sets intercept of the function. Defualt sets intercept to zero
    """
    def __init__(self, negative_slope=0.05, offset=-0.6931):
        super(LeakySoftplus, self).__init__()
        self.negative_slope = negative_slope
        alpha = negative_slope
        offset = (1 - alpha) * offset

        sp = torch.nn.Softplus()
        self.leaky_softplus = lambda x: alpha * x + (1-alpha) * sp(x) + offset

    def forward(self, xs):
        return self.leaky_softplus(xs)


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
