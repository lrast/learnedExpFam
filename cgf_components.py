import torch
import numpy as np

from torch import nn
from scipy.stats import rv_continuous, chi


class normal_radius_uniform_angle(rv_continuous):
    """ Multivariate distribution that is angularly uniform, but has
        a chi(1) marginal distribution along the radius, like a
        normal distribution

        Prevents the samples from accumulating at large radii in high
        dimensions
    """
    def __init__(self, n_dims, stdev):
        self.n_dims = n_dims
        self.scale_dist = chi(1, scale=stdev)

    def rvs(self, n_samples):
        angles = np.random.randn(n_samples, self.n_dims)
        angles = angles / np.linalg.norm(angles, axis=1)[:, None]

        radii = self.scale_dist.rvs((n_samples, 1))
        return torch.tensor(angles * radii, dtype=torch.float32)


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


def n_orthogonal_directions(n, dim):
    """ Find  n directions in dim dimensions that are close to
        orthogonal to each other.
    """
    def to_minimize(x):
        x_normalized = x / torch.norm(x, dim=1)[:, None]
        return torch.abs(x_normalized @ x_normalized.T).sum() - n
    
    points = torch.randn(n, dim)
    points = points / points.norm()
    points = nn.Parameter(points)
    
    optimizer = torch.optim.SGD((points,), lr=1E-3)
    for step in range(200):
        curr_val = optimizer.param_groups[0]['params'][0]
        optimizer.zero_grad()
            
        out = to_minimize(curr_val)
        out.backward()
        optimizer.step()

    outs = optimizer.param_groups[0]['params'][0]
    return outs / torch.norm(outs, dim=1)[:, None]


def n_distant_points(n, dim):
    """ find n points on a dim dimensional sphere that are as far
        from each other as possible
    """
    def to_maximize(x):
        x_normalized = x / torch.norm(x, dim=1)[:, None]
        return torch.cdist(x_normalized, x_normalized, p=2).sum() / 2
    
    points = torch.randn(n, dim)
    points = points / points.norm()
    points = nn.Parameter(points)
    
    optimizer = torch.optim.Adam((points,), lr=1E0, maximize=True)
    for step in range(200):
        curr_val = optimizer.param_groups[0]['params'][0]
        optimizer.zero_grad()
            
        out = to_maximize(curr_val)
        out.backward()
        optimizer.step()

    outs = optimizer.param_groups[0]['params'][0]
    return (outs / torch.norm(outs, dim=1)[:, None]).detach()
    
