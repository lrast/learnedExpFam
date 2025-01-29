import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn
from torch.nn import Parameter

from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.func import vmap, jacrev, hessian

from convex_initialization import ConvexLinear
from cgf_components import normal_radius_uniform_angle, LeakySoftplus


class CGF_ICNN(pl.LightningModule):
    """ Input convex neural network implementation for learning 
        cumulant generating functions

        To do: remove resampling when I'm sure that I'm not going to use it anymore
    """
    def __init__(self, data_to_model, sample_theta=normal_radius_uniform_angle,
                 **kwargs):
        super(CGF_ICNN, self).__init__()

        self.sample_theta = sample_theta
        self.data = data_to_model.detach().clone()
        input_dim = data_to_model.shape[1]

        hyperparameterValues = {
            # seed
            'seed': torch.random.seed(),
            # architecture
            'input_dim': input_dim,
            'negative_slope': 0.05,
            'offset': -0.6931,
            'hidden_size': 200,

            # training
            'lr': 1E-3,
            'batchsize': 128,
            'patience': 100,
            'bias_noise': 0.5,
            'max_epochs': 10000,

            # training data
            'numsamples': 20000,
            'variance': 3.,
            'resample_freq': None,

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

        self.model = nn.Sequential(
                nn.Linear(self.hparams.input_dim, self.hparams.hidden_size),
                LeakySoftplus(self.hparams.negative_slope, self.hparams.offset),
                ConvexLinear(self.hparams.hidden_size, self.hparams.hidden_size,
                             bias_noise=self.hparams.bias_noise),
                LeakySoftplus(self.hparams.negative_slope, self.hparams.offset),
                ConvexLinear(self.hparams.hidden_size, self.hparams.hidden_size,
                             bias_noise=self.hparams.bias_noise),
                LeakySoftplus(self.hparams.negative_slope, self.hparams.offset),
                ConvexLinear(self.hparams.hidden_size, 1,
                             bias_noise=self.hparams.bias_noise)
            )

        self.lossFn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

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
    def dual_opt(self, p, optim_method=torch.optim.Adam, **optkwargs):
        """
            Solve the dual optimization problem.

            Note that issues may arise if the desired slope is not achieved
            by the CGF
        """
        def to_minimize(x):
            return -(torch.einsum('Nk, Nk -> N', p, x) - self.fwd_cpu(x).squeeze())

        input_val = Parameter(torch.zeros(p.shape))
        opt_params = {**{'lr': 1E-3}, **optkwargs}
        optimizer = optim_method((input_val,), **opt_params)
        
        for step in range(500):
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

        if self.hparams.resample_freq and \
           ((self.current_epoch + 1) % self.hparams.resample_freq == 0):
            # resample the training and validation data
            print('resample', self.current_epoch+1, self.hparams.resample_freq)
            self.setup()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # Manage data
    def empirical_CGF(self, ts, data=None):
        """ compute empirical CGF """
        if data is None:
            data = self.data

        num_points = torch.tensor(data.shape[0])

        def find_eCGF(params):
            return torch.logsumexp(data @ params.T, 0) - torch.log(num_points)

        # batch out the computations if necessary
        max_intermediate_elements = int(1E10) / data.element_size()
        max_rows = int(max_intermediate_elements / num_points)

        if ts.shape[0] < max_rows:
            outs = find_eCGF(ts)
        else:
            t_splits = torch.split(ts, max_rows)
            outs_splits = list(map(find_eCGF, t_splits))
            outs = torch.concat(outs_splits)

        return outs

    def setup(self, stage=None):
        """ Generate samples of the empirical moment generating function
            for our dataset.
        """
        N_dims = self.data.shape[1]

        if self.hparams.variance is None:
            raise NotImplementedError('Automatic variance determination is not implemented')
        else:
            variance = self.hparams.variance

        self.thetas = self.sample_theta(N_dims, variance**0.5
                                        ).rvs(self.hparams.numsamples)

        self.CGFs = self.empirical_CGF(self.thetas)[:, None]

        # set up the datasets
        full_dataset = TensorDataset(self.thetas, self.CGFs)
        self.train_split, self.val_split = random_split(
            full_dataset, (0.95, 0.05))

    def train_dataloader(self):
        return DataLoader(self.train_split,
                          batch_size=self.hparams.batchsize,
                          shuffle=True,
                          num_workers=10,
                          persistent_workers=True
                          )

    def val_dataloader(self):
        val_size = len(self.val_split)
        return DataLoader(self.val_split, batch_size=val_size,
                          num_workers=2,
                          persistent_workers=True
                          )


class ConditionalCGF(CGF_ICNN):
    """ConditionalCGF: CGF network subclasses that takes into account the labels
       while fitting the CGF
    """
    def __init__(self, dataset_to_model, sample_theta=normal_radius_uniform_angle,
                 **kwargs):
        # split data and targets
        data, targets = dataset_to_model[:]
        super(ConditionalCGF, self).__init__(data, sample_theta,
                                             **kwargs)

        self.targets = targets

        # subclass specific hyperparameters
        hyperparameters = {'parameter_radius': 1.0}
        hyperparameters.update((k, kwargs[k])
                               for k in hyperparameters.keys() & kwargs.keys())
        self.save_hyperparameters(hyperparameters)

    # training with a two input dataloader
    def training_step(self, batch, batchidx):
        parameters, ts, targets = batch
        conditionalCGF = self.forward(parameters + ts) - self.forward(parameters)

        loss = self.lossFn(conditionalCGF, targets)
        self.log('Train Loss', loss)
        return loss

    def validation_step(self, batch, batchidx):
        parameters, ts, targets = batch
        conditionalCGF = self.forward(parameters + ts) - self.forward(parameters)

        loss = self.lossFn(conditionalCGF, targets)
        self.log('Val Loss', loss)

    # parameter functions
    def make_parameters(self, targets):
        """ Describes how the parameters of the network vary with the target.

            For now, this is an input.

            Assume discrete target values
        """

        targets = targets.squeeze()
        """
        hardcoded = torch.tensor([[-1.0000e+00,  8.7423e-08],
                                    [-8.6603e-01, -5.0000e-01],
                                    [-5.0000e-01, -8.6603e-01],
                                    [-4.3711e-08, -1.0000e+00],
                                    [ 5.0000e-01, -8.6603e-01],
                                    [ 8.6603e-01, -5.0000e-01],
                                    [ 1.0000e+00,  0.0000e+00],
                                    [ 8.6603e-01,  5.0000e-01],
                                    [ 5.0000e-01,  8.6603e-01],
                                    [-4.3711e-08,  1.0000e+00],
                                    [-5.0000e-01,  8.6603e-01],
                                    [-8.6603e-01,  5.0000e-01]])
        angles = hardcoded[targets]
        """

        radius = self.hparams.parameter_radius
        angles = self.parameter_angles[targets]

        return radius * angles

    def sample_dual_training(self):
        """ Samples values of the dual variable to be used in training 
            the CGF network
        """
        N_dims = self.hparams.input_dim

        if self.hparams.variance is None:
            raise NotImplementedError('Automatic variance determination is not implemented')
        else:
            variance = self.hparams.variance

        thetas = self.sample_theta(N_dims, variance**0.5
                                   ).rvs(self.hparams.numsamples)

        if N_dims == 1:  # edge case
            thetas = thetas[:, None]

        return thetas

    def setup(self, stage=None):
        """ Generate samples of the empirical moment generating function
            for our dataset.

            Currently: align parameters with mean
        """
        # debug code
        try:
            a = self.train_split
            return
        except AttributeError:
            pass

        targets = self.targets
        possible_targets = torch.unique(targets, dim=0)

        # setup the parameter embedding
        self.parameter_angles = torch.zeros(len(possible_targets),
                                            self.hparams.input_dim)

        all_CGF_values = []

        for target in possible_targets:
            inds = (targets == target).squeeze()
            data_select = self.data[inds]

            ts = self.sample_dual_training()
            CGF_values = self.empirical_CGF(ts, data=data_select)[:, None]

            angle = data_select.mean(0)
            angle = angle / torch.norm(angle)
            self.parameter_angles[target] = angle
            parameter = self.hparams.parameter_radius * angle
            parameters = parameter.repeat(ts.shape[0], 1)

            all_CGF_values.append(torch.cat([parameters, ts, CGF_values], dim=1))

        full_train = torch.cat(all_CGF_values, dim=0)

        CGFs = full_train[:, -1:]
        parameters = full_train[:, 0:self.hparams.input_dim]
        ts = full_train[:, self.hparams.input_dim:2*self.hparams.input_dim]

        # set up the datasets
        full_dataset = TensorDataset(parameters, ts, CGFs)
        self.train_split, self.val_split = random_split(
            full_dataset, (0.8, 0.2))
