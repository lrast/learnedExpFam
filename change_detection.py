# minimum viable product

import torch
import warnings

import numpy as np

from scipy.stats import multivariate_normal
from scipy.stats import chi2


class ScoreTest(object):
    """ Detect changes using the CGF alone via the score function test
    """
    def __init__(self, cgf_model, n_samples, pvalue):
        super(ScoreTest, self).__init__()
        self.cgf_model = cgf_model
        self.n_samples = n_samples
        self.pvalue = pvalue

    def is_different(self, data):
        """ Determine whether this data represents a significant deviation
            from the baseline theta = 0
        """
        n_samples, n_dims = data.shape

        model_mean = self.cgf_model.jac(torch.zeros(1, 2))[0].detach()
        model_cov = self.cgf_model.hess(torch.zeros(1, 2))[0].detach()

        emp_mean = data.mean(0)

        score = (1./n_samples) * (emp_mean - model_mean).T @ \
            torch.linalg.inv(model_cov) @ (emp_mean - model_mean)

        return chi2(n_dims).cdf(score) > self.pvalue


class RateFunctionTest(object):
    """ Detects changes in the mean by determining whether the it differs
        significantly from cgf_model values

        Uses level curves of the rate function deliminate significance regions
    """
    def __init__(self, cgf_model, n_samples, pvalue):
        super(RateFunctionTest, self).__init__()
        self.cgf_model = cgf_model
        self.n_samples = n_samples

        self.set_threashold(pvalue)

    def limiting_density(self, mus):
        """ Model based asymptotic pdf of the mean """
        d = mus.shape[1]

        thetas, Is = self.cgf_model.dual_opt(mus)
        dets = torch.det(self.cgf_model.hess(thetas))
        
        log_density = (d/2) * np.log(self.n_samples / (2 * torch.pi)) \
            - 0.5*torch.log(dets) - self.n_samples*Is

        return torch.exp(log_density).detach()

    def set_threashold(self, p_value):
        """ Find the threashold value for the rate function that gives the 
        desired p-value.
        """
        n_samples = self.n_samples

        def within_rate_contour(xs, value):
            """ are the point inside the rate function contour? """
            I_vals = self.cgf_model.dual_function(xs)

            return (I_vals < value).double()

        # importance sampling distribution
        mean = self.cgf_model.jac(torch.zeros(1, 2))[0].detach()
        cov = self.cgf_model.hess(torch.zeros(1, 2))[0].detach()
        sample_dist = multivariate_normal(mean=mean, cov=(1./n_samples)*cov)

        def empirical_p(threashold):
            func = lambda x: within_rate_contour(x, threashold)
            return importance_sample(func, self.limiting_density, sample_dist).item()

        self.threashold = binary_search_threashold(empirical_p, p_value)

    def is_different(self, data):
        """ Determine whether this data represents a significant deviation """
        if data.shape[0] != self.n_samples:
            raise Exception("Incorrect number of samples")

        mean = data.mean(0)
        rate_value = self.cgf_model.dual_function(mean[None, :])

        return rate_value > self.threashold


# Utilities
#
#

def importance_sample(func, target_pdf, sample_dist, N=10000,
                      weight_norm=True, resample=False):
    """
        Fully feaured importance sampling, for p-threashold evaluation
    """
    sample_pdf = sample_dist.pdf
    sample_rvs = sample_dist.rvs

    samples = torch.as_tensor(sample_rvs(N), dtype=torch.float)

    importance_weights = target_pdf(samples) / torch.as_tensor(sample_pdf(samples))
    w_total = importance_weights.sum()

    if w_total / N < 0.1:
        warnings.warn(f"Low effective sample number {w_total / N}")

    if weight_norm or resample:
        importance_weights = importance_weights / w_total
    else:
        importance_weights = importance_weights / N

    if resample:
        inds = np.random.choice(np.arange(0, N), size=N, p=importance_weights)

        samples = samples[inds]
        fvals = func(samples)
        importance_weights = np.ones(N) / N

    fvals = func(samples)

    return (fvals * importance_weights).sum()


def binary_search_threashold(empirical_p, pval, guess=1, delta=0.01,  Nmax=1000):
    """ Binary search for threashold of interest.

        Currently doesn't take into account noise in the data,
        which might lead to incorrect updates
    """
    mag_bounds = [-float('inf'), float('inf')]
    p_bounds = [-float('inf'), float('inf')]

    threashold = guess
    
    for i in range(Nmax):
        frac = empirical_p(threashold)
        if frac > pval and frac < p_bounds[1]:
            mag_bounds[1] = threashold
            p_bounds[1] = frac
            
        elif frac < pval and frac > p_bounds[0]:
            mag_bounds[0] = threashold
            p_bounds[0] = frac

        # termination criterion
        if p_bounds[1] - p_bounds[0] < delta:
            return 0.5*(mag_bounds[0] + mag_bounds[1])

        # next step:
        if mag_bounds[0] == -float('inf'):
            threashold = threashold / 2
        elif mag_bounds[1] == float('inf'):
            threashold = 2 * threashold
        else:
            threashold = 0.5*(mag_bounds[0] + mag_bounds[1])

    raise Exception("Convergence failure")
