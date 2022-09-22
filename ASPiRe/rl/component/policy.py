import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.distributions import Distribution as TorchDistribution
from typing import Dict, Iterable, Optional, Tuple, Union, List, Type, NamedTuple
import torch as th
from ASPiRe.modules.distributions import MultivariateDiagonalNormal, TanhNormal
from ASPiRe.modules.network import create_mlp
from ASPiRe.rl.component.preprocessor import preprocessor

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class PolicyOuput(NamedTuple):
    actions: torch.tensor
    logprob: torch.tensor
    distribution: TorchDistribution


class policy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        pass


class actor(policy):

    def __init__(self,
                 net_arch=[128, 128],
                 max_action_range=2,
                 action_dim=10,
                 preprocessor: preprocessor = None,
                 tanh_squash_porb=True,
                 use_batch_norm=False,
                 device=None):
        super().__init__()
        self.preprocessor = preprocessor
        input_dim = self.preprocessor.output_dim
        latent_policy = create_mlp(input_dim=input_dim, output_dim=-1, net_arch=net_arch, use_batch_norm=use_batch_norm)
        self.latent_policy = nn.Sequential(*latent_policy)

        self.mu = nn.Linear(net_arch[-1], action_dim)
        self.log_std = nn.Linear(net_arch[-1], action_dim)

        self.max_action_range = max_action_range
        self.tanh_squash_porb = tanh_squash_porb
        self.device = device
        self.to(device)

    def forward(self, observation):
        feature = self.preprocessor(observation)
        policy_latent = self.latent_policy(feature)
        mu = self.mu(policy_latent)
        log_std = self.log_std(policy_latent)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        policy_distribution = MultivariateDiagonalNormal(mu, log_std.exp())
        pi_action = policy_distribution.rsample()
        log_pi = policy_distribution.log_prob(pi_action)
        if self.tanh_squash_porb:
            pi_action, log_pi = self._tanh_squash_output(pi_action, log_pi)
        return PolicyOuput(pi_action, log_pi, policy_distribution)

    def _tanh_squash_output(self, action, log_prob):
        """Passes continuous output through a tanh function to constrain action range, adjusts log_prob."""
        action_new = self.max_action_range * torch.tanh(action)
        log_prob_update = np.log(self.max_action_range) + 2 * (np.log(2.) - action - torch.nn.functional.softplus(
            -2. * action)).sum(dim=-1)  # maybe more stable version from Youngwoon Lee
        return action_new, log_prob - log_prob_update


class critic(nn.Module):

    def __init__(self,
                 net_arch: List[int],
                 action_dim=10,
                 n_critics: int = 2,
                 preprocessor: preprocessor = None,
                 share_preprocessor: bool = True):
        super().__init__()
        self.preprocessor = preprocessor
        self.share_preprocessor = share_preprocessor
        self.n_critics = n_critics
        self.q_networks = []
        input_dim = self.preprocessor.output_dim
        for idx in range(n_critics):
            q_net = create_mlp(
                input_dim=input_dim + action_dim,
                output_dim=1,
                net_arch=net_arch,
            )
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

        # self.to(device)

    def forward(self, observation, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        with torch.set_grad_enabled(not self.share_preprocessor):
            feature = self.preprocessor(observation)
        qvalue_input = th.cat([feature, actions], dim=1)
        qvalues = [q_net(qvalue_input) for q_net in self.q_networks]
        return qvalues
