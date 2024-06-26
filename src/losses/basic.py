import torch
from torch.distributions import Distribution


def negative_gaussian_logpdf(inputs, dist: Distribution, reduction=None):
    """Gaussian log-density.
    Args:
        inputs: Inputs.
        dist: distribution.
        reduction: Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".
    Returns:
        Log-density.
    """
    logp = dist.log_prob(inputs)

    if not reduction:
        return -logp
    elif reduction == 'sum':
        return -torch.sum(logp)
    elif reduction == 'mean':
        return -torch.mean(logp)
    elif reduction == 'batched_mean':
        return -torch.mean(torch.sum(logp, 1))
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')


def negative_kl_div(prior_dist, posterior_dist, reduction=None):
    kl = torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
    if not reduction:
        return -kl
    elif reduction == 'sum':
        return -torch.sum(kl)
    elif reduction == 'mean':
        return -torch.mean(kl)
    elif reduction == 'batched_mean':
        return -torch.mean(torch.sum(kl, 1))
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')

# def negative_kl_div(prior_mu, prior_sigma, posterior_mu, posterior_sigma, reduction=None):
#     prior_dist = torch.distributions.Normal(loc=prior_mu, scale=prior_sigma)
#     posterior_dist = torch.distributions.Normal(loc=posterior_mu, scale=posterior_sigma)
#     kl = torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
#     if not reduction:
#         return -kl
#     elif reduction == 'sum':
#         return -torch.sum(kl)
#     elif reduction == 'mean':
#         return -torch.mean(kl)
#     elif reduction == 'batched_mean':
#         return -torch.mean(torch.sum(kl, 1))
#     else:
#         raise RuntimeError(f'Unknown reduction "{reduction}".')
