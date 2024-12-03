"""
Adapted from the the repository:
    https://github.com/ml-jku/convex-init
for the paper:
    Hoedt & Klambauer Principled Weight Initialisation for Input-Convex Neural Networks 2023
    https://neurips.cc/virtual/2023/poster/70408
 @inproceedings{hoedt2023principled,
   title     = {Principled Weight Initialisation for Input-Convex Neural Networks},
   author    = {Hoedt, Pieter-Jan and Klambauer, G{\"u}nter Klambauer},
   booktitle = {Thirty-seventh Conference on Neural Information Processing Systems},
   year      = {2023},
   url       = {https://openreview.net/forum?id=pWZ97hUQtQ}
 }

See the license ./LICENSE_convex_init
"""

from typing import Concatenate
from scipy.stats import Covariance
import torch
import math

from torch import nn
from abc import ABC, abstractmethod
from inspect import getfullargspec as argspec


class Positivity(ABC):
    """ Interface for function that makes weights positive. """

    @abstractmethod
    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        """ Transform raw weight to positive weight. """
        ...

    def inverse_transform(self, pos_weight: torch.Tensor) -> torch.Tensor:
        """ Transform positive weight to raw weight before transform. """
        return self.__call__(pos_weight)


class LazyClippedPositivity(Positivity):
    """
    Make weights positive by clipping negative weights after each update.

    References
    ----------
    Amos et al. (2017)
        Input-Convex Neural Networks.
    """

    def __call__(self, weight):
        with torch.no_grad():
            weight.clamp_(0)

        return weight


class ClippedPositivity(Positivity):
    """
    Make weights positive by using applying ReLU during forward pass.
    """

    def __call__(self, weight):
        return torch.relu(weight)


class ConvexLinear(nn.Linear):
    """Linear layer with positive weights."""

    def __init__(self, *args, positivity: Positivity = ClippedPositivity(),
                 convex_init=True, **kwargs):
        if positivity is None:
            raise TypeError("positivity must be given as kwarg for convex layer")

        self.positivity = positivity
        initialize_kwargs = {name: kwargs.pop(name)
                             for name in argspec(ConvexInitialiser.__init__).args[1:]
                             if name in kwargs
                             }

        super().__init__(*args, **kwargs)

        if convex_init:
            ConvexInitialiser(**initialize_kwargs)(*self.parameters())

    def forward(self, x):
        return torch.nn.functional.linear(x, self.positivity(self.weight), self.bias)


class ConvexInitialiser:
    """
    Initialisation method for input-convex networks.

    Parameters
    ----------
    var : float, optional
        The target variance fixed point.
        Should be a positive number.
    corr : float, optional
        The target correlation fixed point.
        Should be a value between -1 and 1, but typically positive.
    bias_noise : float, optional
        The fraction of variance to originate from the bias parameters.
        Should be a value between 0 and 1
    alpha : float, optional
        Scaling parameter for leaky ReLU.
        Should be a positive number.

    Examples
    --------
    Default initialisation

    >>> icnn = torch.nn.Sequential(
    ...     torch.nn.Linear(200, 400),
    ...     torch.nn.ReLU(),
    ...     ConvexLinear(400, 300),
    ... )
    >>> torch.nn.init.kaiming_uniform_(icnn[0].weight, nonlinearity="linear")
    >>> torch.nn.init.zeros_(icnn[0].bias)
    >>> convex_init = ConvexInitialiser()
    >>> w1, b1 = icnn[1].parameters()
    >>> convex_init(w1, b1)
    >>> assert torch.all(w1 >= 0) and torch.isclose(b1.var(), torch.zeros(1))

    Initialisation with random bias parameters

    >>> convex_bias_init = ConvexInitialiser(bias_noise=0.5)
    >>> convex_bias_init(w1, b1)
    >>> assert torch.all(w1 >= 0) and b1.var() > 0
    """

    @staticmethod
    @torch.no_grad()
    def init_log_normal_(weight: torch.Tensor, mean_sq: float, var: float) -> torch.Tensor:
        """
        Initialise weights with samples from a log-normal distribution.

        Parameters
        ----------
        weight : torch.Tensor
            The parameter to be initialised.
        mean_sq : float
            The squared mean of the normal distribution underlying the log-normal.
        var : float
            The variance of the normal distribution underlying the log-normal.

        Returns
        -------
        weight : torch.Tensor
            A reference to the inputs that have been modified in-place.
        """
        log_mom2 = math.log(mean_sq + var)
        log_mean = math.log(mean_sq) - log_mom2 / 2.
        log_var = log_mom2 - math.log(mean_sq)
        return torch.nn.init.normal_(weight, log_mean, log_var ** .5).exp_()

    def __init__(self, var: float = 1., corr: float = 0.5,
                 bias_noise: float = 0., alpha: float = 0.):
        self.target_var = var
        self.target_corr = corr
        self.bias_noise = bias_noise
        self.relu_scale = 2. / (1. + alpha ** 2)

    def __call__(self, weight: torch.Tensor, bias: torch.Tensor):
        if bias is None:
            raise ValueError("Principled Initialisation for ICNNs requires bias parameter")

        fan_in = torch.nn.init._calculate_correct_fan(weight, "fan_in")
        weight_dist, bias_dist = self.compute_parameters(fan_in)
        weight_mean_sq, weight_var = weight_dist
        self.init_log_normal_(weight, weight_mean_sq, weight_var)

        bias_mean, bias_var = bias_dist
        torch.nn.init.normal_(bias, bias_mean, bias_var ** .5)

    def compute_parameters(self, fan_in: int) -> tuple[
        tuple[float, float], tuple[float, float] | None
    ]:
        """
        Compute the distribution parameters for the initialisation.

        Parameters
        ----------
        fan_in : int
            Number of incoming connections.

        Returns
        -------
        (weight_mean_sq, weight_var) : tuple of 2 float
            The squared mean and variance for weight parameters.
        (bias_mean, bias_var): tuple of 2 float, optional
            The mean and variance for the bias parameters.
            If `no_bias` is `True`, `None` is returned instead.
        """
        target_mean_sq = self.target_corr / self.corr_func(fan_in)
        target_variance = self.relu_scale * (1. - self.target_corr) / fan_in

        shift = fan_in * (target_mean_sq * self.target_var / (2 * math.pi)) ** .5
        bias_var = 0.
        if self.bias_noise > 0.:
            target_variance *= (1 - self.bias_noise)
            bias_var = self.bias_noise * (1. - self.target_corr) * self.target_var

        return (target_mean_sq, target_variance), (-shift, bias_var)

    def corr_func(self, fan_in: int) -> float:
        """ Helper function for correlation """
        rho = self.target_corr
        mix_mom = (1 - rho ** 2) ** .5 + rho * math.acos(-rho)
        return fan_in * (math.pi - fan_in + (fan_in - 1) * mix_mom) / (2 * math.pi)
