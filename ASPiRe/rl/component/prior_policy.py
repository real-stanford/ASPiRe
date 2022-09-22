import torch
from typing import Dict, Iterable, Optional, Tuple, Union, List, Type, NamedTuple

from ASPiRe.modules.distributions import MultivariateDiagonalNormal, ScaledMultivariateDiagonalNormal
from ASPiRe.rl.component.policy import actor, PolicyOuput
from ASPiRe.rl.component.preprocessor import preprocessor
from ASPiRe.rl.utility.helper import check_shape, no_batchnorm_update
from ASPiRe.rl.utility.distribution import concat_normals_param, mc_kl_divergence
from gym import Env


class PriorOuput(NamedTuple):
    distributions: List[ScaledMultivariateDiagonalNormal]
    param: torch.Tensor


class prior_actor(actor):

    def __init__(self,
                 net_arch=[128, 128],
                 max_action_range=2,
                 tanh_squash_porb=True,
                 action_dim=10,
                 preprocessor: preprocessor = None,
                 priors=None,
                 execute_prior=False,
                 execute_prior_index=0,
                 use_batch_norm=False,
                 freeze_batch_norm=False,
                 device=None):
        super().__init__(net_arch=net_arch,
                         max_action_range=max_action_range,
                         tanh_squash_porb=tanh_squash_porb,
                         action_dim=action_dim,
                         preprocessor=preprocessor,
                         use_batch_norm=use_batch_norm,
                         device=device)

        self.priors = priors
        self.execute_prior = execute_prior
        self.execute_prior_index = execute_prior_index
        self.freeze_batch_norm = freeze_batch_norm

    def get_prior(self, observations):
        with torch.no_grad():
            with no_batchnorm_update(self):
                priors = [prior(self.get_prior_input(observations, i)) for i, prior in enumerate(self.priors)]
                prior_params = concat_normals_param(priors)

                return PriorOuput(priors, prior_params)

    def get_prior_input(self, observation, prior_index):

        return observation['vector']

    def forward(self, observation):
        if self.execute_prior:
            return self.execute_prior_skill(observation, self.execute_prior_index)
        elif self.freeze_batch_norm:
            with no_batchnorm_update(self):
                return super().forward(observation)
        else:
            return super().forward(observation)

    def execute_prior_skill(self, observation, execute_prior_index):
        priors = self.get_prior(observation).distributions
        execute_priors = priors[execute_prior_index]
        pi_action = execute_priors.sample()
        pi_action = torch.clip(pi_action, -self.max_action_range, self.max_action_range)
        log_pi = execute_priors.log_prob(pi_action)
        return PolicyOuput(pi_action, log_pi, execute_priors)
