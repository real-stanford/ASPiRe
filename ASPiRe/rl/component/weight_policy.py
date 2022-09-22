from typing import Dict, Iterable, Optional, Tuple, Union, List, Type, NamedTuple
from torch.distributions import Distribution as TorchDistribution

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

from ASPiRe.rl.component.preprocessor import preprocessor, vector_preprocessor
from ASPiRe.rl.component.policy import actor, critic, PolicyOuput, LOG_STD_MAX, LOG_STD_MIN
from ASPiRe.modules.network import create_mlp
from ASPiRe.modules.distributions import MultivariateDiagonalNormal, TanhNormal
from ASPiRe.rl.utility.helper import check_shape
import numpy as np

MIN_MU = 1e-3


class WeightPolicyOuput(NamedTuple):
    weights: torch.tensor
    logprob: torch.tensor
    precision: torch.tensor
    mu: torch.tensor
    distribution: TorchDistribution


class SoftMaxWeightActor(nn.Module):

    def __init__(self,
                 net_arch: list = [128, 128],
                 weight_dim: int = None,
                 preprocessor: preprocessor = None,
                 temperature: int = 1,
                 use_batch_norm: bool = False,
                 device: int = 0):
        super().__init__()
        self.weight_dim = weight_dim
        self.preprocessor = preprocessor
        input_dim = self.preprocessor.output_dim
        latent_policy = create_mlp(input_dim=input_dim, output_dim=-1, net_arch=net_arch, use_batch_norm=use_batch_norm)
        self.latent_policy = nn.Sequential(*latent_policy)

        self.mu = nn.Sequential(nn.Linear(net_arch[-1], self.weight_dim))

        self.device = device
        self.to(self.device)
        self.temperature = temperature

    def forward(self, observation):
        feature = self.preprocessor(observation)
        policy_latent = self.latent_policy(feature)
        mu = self.mu(policy_latent)
        weight = F.softmax(mu / self.temperature, dim=1)
        return WeightPolicyOuput(weight, None, 0, weight, None)


class WeightDecoder(actor):
    # w -> a
    def __init__(self, net_arch: list = [128, 128], action_dim=10, preprocessor: preprocessor = None, device=None):
        super().__init__(net_arch=net_arch, action_dim=action_dim, preprocessor=preprocessor, device=device)

    def forward(self, weights: torch.Tensor, priors: torch.Tensor):
        policy_latent = self.latent_policy(torch.cat([weights, priors], dim=1))
        mu = self.mu(policy_latent)
        log_std = self.log_std(policy_latent)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        generated_distribution = MultivariateDiagonalNormal(mu, log_std.exp())
        pi_action = generated_distribution.rsample()
        log_pi = generated_distribution.log_prob(pi_action)
        return PolicyOuput(pi_action, log_pi, generated_distribution)


class AdaptiveWeight(nn.Module):

    def __init__(self,
                 net_arch: list = [128, 128],
                 weight_dim: int = None,
                 action_dim: int = None,
                 preprocessor: preprocessor = None,
                 temperature: int = 1,
                 update_start: int = 10000,
                 policy: str = "softmax",
                 use_batch_norm: bool = False,
                 min_coef=1e-3,
                 max_updates=3e4,
                 device: int = 0) -> None:
        super().__init__()
        self.weight_dim = weight_dim
        self.policy = policy
        if self.policy == 'softmax':
            self.weight_actor = SoftMaxWeightActor(net_arch=net_arch,
                                                   weight_dim=weight_dim,
                                                   preprocessor=preprocessor,
                                                   temperature=temperature,
                                                   use_batch_norm=use_batch_norm,
                                                   device=device)
        else:
            raise NotImplementedError

        self.weight_critic = critic(net_arch=[128] * 6,
                                    action_dim=weight_dim,
                                    preprocessor=self.weight_actor.preprocessor).to(device)

        self.weight_critic_target = critic(net_arch=[128] * 6,
                                           action_dim=weight_dim,
                                           preprocessor=self.weight_actor.preprocessor).to(device)

        self.weight_decoder = WeightDecoder(net_arch=net_arch,
                                            action_dim=action_dim,
                                            preprocessor=vector_preprocessor(input_dim=weight_dim +
                                                                             weight_dim * action_dim * 2,
                                                                             fc_net_arch=[],
                                                                             output_dim=-1),
                                            device=device)

        self.update_start = update_start
        self.min_coef = min_coef
        self.max_updates = max_updates
        self.to(device)

    def lamda_annealing(self, n_updates):
        max_updates = self.max_updates
        max_value = 1
        min_value = self.min_coef
        return min_value + (max_value - min_value) * max(max_updates - n_updates, 0) / max_updates

    @staticmethod
    def n_ints_summing_to_v(n, v):
        return np.random.multinomial(v, np.ones((n)) / float(n))
