import torch
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from ASPiRe.modules.distributions import MultivariateDiagonalNormal

MAX_KL = 100


def mc_kl_divergence(p, q, n_samples=10, theta=1):
    """Computes monte-carlo estimate of KL divergence. n_samples: how many samples are used for the estimate."""
    samples = [p.rsample() for _ in range(n_samples)]
    return torch.stack([theta * p.log_prob(x) - q.log_prob(x) for x in samples], dim=1).mean(dim=1)


def mc_nll_divergence(a, q):
    """Computes monte-carlo estimate of negative log prob. n_samples: how many samples are used for the estimate."""

    return -q.log_prob(a)


def compute_prior_divergence(distribution=None,
                             priors: List = None,
                             n_samples: int = 10,
                             theta: int = 1,
                             analytic=False):
    if analytic:
        kl_divs = [torch.distributions.kl_divergence(distribution.distribution, prior.distribution) for prior in priors]
    else:
        kl_divs = [mc_kl_divergence(distribution, prior, n_samples, theta) for prior in priors]

    kl_divs = torch.stack(kl_divs, dim=-1)
    kl_divs = torch.clamp(kl_divs, min=0, max=MAX_KL)

    return kl_divs


def compute_prior_nll(actions=None, priors: List = None):

    nlls = [mc_nll_divergence(
        actions,
        prior,
    ) for prior in priors]

    nlls = torch.stack(nlls, dim=-1)

    return nlls


def concat_normals_param(normal_distributions: List[MultivariateDiagonalNormal]):
    with torch.no_grad():
        param_list = []
        for d in normal_distributions:
            param_list.append(d.mean)
            param_list.append(d.stddev)

        param_list = torch.cat(param_list, dim=1)

        return param_list


def scale_normal_distribution(normal_distribution: MultivariateDiagonalNormal, scale: int = 1):
    ret_mean = normal_distribution.mean * scale
    ret_stddev = normal_distribution.stddev * scale
    ret_normal = MultivariateDiagonalNormal(ret_mean, ret_stddev)

    return ret_normal
