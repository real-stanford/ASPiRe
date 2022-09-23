import pickle
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import polyak_update
from ASPiRe.rl.utility.distribution import compute_prior_nll, mc_kl_divergence
from ASPiRe.rl.component.policy import PolicyOuput, critic
from ASPiRe.rl.component.prior_policy import PriorOuput, prior_actor
from ASPiRe.rl.component.replay_buffer import ReplayBufferSamples
from ASPiRe.rl.component.weight_policy import AdaptiveWeight, WeightPolicyOuput
from ASPiRe.rl.utility.helper import check_shape, dict_to_torch, to_torch, no_batchnorm_update
from ASPiRe.rl.utility.distribution import compute_prior_divergence
import wandb
import time
import os
import gym

N_SAMPLES = 20
MAX_WEIGHTED_EXP_ADV = 10
MIN_WEIGHTED_EXP_ADV = -10

MAX_ADVANTAGE = 1
MIN_ADVANTAGE = -1

MAX_LOGPROB = 10
MIN_LOGPROB = -10

CLIP_VALUE = 0.1


class CompositeSAC():

    def __init__(self,
                 policy: prior_actor = None,
                 weight_policy: AdaptiveWeight = None,
                 env: GymEnv = None,
                 learning_rate: float = 3e-4,
                 critic_lr: float = 3e-4,
                 weight_lr: float = 3e-4,
                 learning_starts: int = 100,
                 batch_size: int = 256,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 train_freq: Union[int, Tuple[int, str]] = 1,
                 gradient_steps: int = 1,
                 replay_buffer_class: ReplayBuffer = None,
                 replay_buffer_size: int = int(1e6),
                 replay_buffer_args=None,
                 target_update_interval: int = 1,
                 target_kl: Union[str, float] = "auto",
                 target_entropy: Union[str, float] = "auto",
                 log: bool = False,
                 normalizer=None,
                 run_name: str = None,
                 checkpoint_frequency: int = 1000,
                 alpha_schedule_fn=None,
                 prior_exploration: bool = False,
                 prior_exploration_steps: int = 0,
                 theta: float = 1.0,
                 lamda: float = 1.0,
                 update_method: str = 'ac',
                 weight_network: str = 'softmax',
                 raw_kl: bool = False,
                 min_entropy_coef=0,
                 min_alpha_coef=0,
                 include_entropy=False,
                 analytic_kl=False,
                 clip_kl_divergence=False,
                 clip_value=100,
                 n_samples=20,
                 clip_critic_gradient=False,
                 critic_clip_value=1,
                 max_action_range=2,
                 policy_reuse=False,
                 eps=0,
                 device: int = 0) -> None:
        self.device = device
        self.env = env
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_lr = weight_lr
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.lamda = lamda
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval

        self.dict_observation = True if type(self.env.observation_space) == gym.spaces.dict.Dict else False
        self.replay_buffer = replay_buffer_class(replay_buffer_size, **replay_buffer_args)

        self.log_alpha = torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
        self.log_beta = torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
        self.log_ent_coef = torch.log(torch.ones(1, device=self.device)).requires_grad_(True)

        self.policy = policy.to(device)
        self.weight_policy = weight_policy.to(device)

        self.critic = critic(preprocessor=policy.preprocessor,
                             net_arch=[256] * 6,
                             share_preprocessor=True,
                             action_dim=10,
                             n_critics=2).to(device)

        self.create_alias()

        self.critic_target = critic(preprocessor=policy.preprocessor, net_arch=[256] * 6, action_dim=10,
                                    n_critics=2).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.train(False)

        self.critic_parameters = [param for name, param in self.critic.named_parameters() if "preprocessor" not in name]
        if not self.policy.execute_prior:
            self.policy_optimizer = Adam(self.policy.parameters(), lr=self.learning_rate)
        self.critic_optimizer = Adam(self.critic_parameters, lr=critic_lr)
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.learning_rate)
        self.min_alpha_coef = min_alpha_coef
        self.beta_optimizer = Adam([self.log_beta], lr=self.learning_rate)
        self.ent_coef_optimizer = Adam([self.log_ent_coef], lr=self.learning_rate)
        self.min_entropy_coef = min_entropy_coef

        self.weight_actor_optimizer = Adam(self.weight_actors.parameters(), lr=self.weight_lr)
        self.weight_critic_parameters = [
            param for name, param in self.weight_critics.named_parameters() if "preprocessor" not in name
        ]
        self.weight_critic_optimizer = Adam(self.weight_critic_parameters, lr=self.weight_lr)

        self.weight_critic_target.load_state_dict(self.weight_critics.state_dict())
        self.weight_critic_target.train(False)

        self.weight_decoder_optimizer = Adam(self.weight_decoder.parameters(), lr=self.weight_lr)

        self.normalizer = normalizer
        self.alpha_schedule_fn = alpha_schedule_fn

        if target_entropy == "auto":
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(target_entropy)

        if target_kl == "auto":
            self.target_kl = np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            self.target_kl = float(target_kl)

        self.num_timesteps = 0
        self.log = log

        self.checkpoint_frequency = checkpoint_frequency
        self.save_dir = os.path.join('../Experiment', run_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.prior_exploration = prior_exploration
        self.prior_exploration_steps = prior_exploration_steps
        self.execute_prior_index = np.random.randint(self.n_priors)
        self.update_method = update_method
        self.raw_kl = raw_kl

        self.theta = theta

        self.weight_network = weight_network
        self.include_entropy = include_entropy
        self.analytic_kl = analytic_kl
        self.clip_kl_divergence = clip_kl_divergence
        self.clip_value = clip_value
        self.n_samples = n_samples

        self.critic_clip_value = critic_clip_value
        self.clip_critic_gradient = clip_critic_gradient
        self.max_action_range = max_action_range
        self.policy_reuse = policy_reuse
        self.eps = eps

    def create_alias(self):
        self.weight_actors = self.weight_policy.weight_actor
        self.weight_critics = self.weight_policy.weight_critic
        self.weight_critic_target = self.weight_policy.weight_critic_target
        self.weight_decoder = self.weight_policy.weight_decoder
        self.n_priors = self.weight_policy.weight_dim

    def update(self, gradient_steps: int, batch_size: int):

        for gradient_step in range(gradient_steps):
            experience_replay = self.replay_buffer.sample(batch_size=batch_size)
            experience_replay = self._preprocess(experience_replay)
            priors_output = self.get_priors(experience_replay.observations)
            policy_output = self._run_policy(experience_replay.observations)
            weight_output = self._run_weight_policy(experience_replay.observations)
            alpha_loss_info = self.update_alpha(priors_output=priors_output,
                                                weight_output=weight_output,
                                                policy_output=policy_output)
            if self.include_entropy:
                ent_coef_loss_info = self.update_ent_coef(policy_output=policy_output)
            else:
                ent_coef_loss_info = {}

            # Compute target q values
            with torch.no_grad():
                next_policy_output = self._run_policy(experience_replay.next_observations)
                next_priors_ouput = self.get_priors(experience_replay.next_observations)
                next_weight_output = self._run_weight_policy(experience_replay.next_observations)
                next_q_value = self.compute_next_q_value(experience_replay.next_observations, next_policy_output,
                                                         next_priors_ouput, next_weight_output)
                check_shape(experience_replay.rewards, [self.batch_size, 1])
                check_shape(experience_replay.dones, [self.batch_size, 1])
                target_q_value = experience_replay.rewards + self.gamma * (1 - experience_replay.dones) * next_q_value
                target_q_value = target_q_value.detach()

            # Compute critic loss
            critic_loss, critic_loss_info = self.compute_critic_loss(experience_replay, target_q_value)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            policy_loss, policy_loss_info = self.compute_policy_loss(experience_replay, policy_output, priors_output,
                                                                     weight_output)

            # Optimize the actor
            self.policy_optimizer.zero_grad()
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            policy_loss.backward()
            self.policy_optimizer.step()

            decoder_loss_info = self.update_weight_decoder(
                experience_replay=experience_replay,
                priors_output=priors_output,
            )

            if self.weight_policy.update_start < self.updates:
                weight_actor_loss_info = self.update_weight_actor(
                    experience_replay=experience_replay,
                    priors_output=priors_output,
                    weight_output=weight_output,
                )
            else:
                weight_actor_loss_info = {}

            # Update the target critic
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        info = {
            **ent_coef_loss_info,
            **critic_loss_info,
            **policy_loss_info,
            **alpha_loss_info,
            **decoder_loss_info,
            **weight_actor_loss_info,
        }
        return info

    def update_ent_coef(self, policy_output):
        ent_coef = torch.exp(self.log_ent_coef)
        target_entropy = self.target_entropy
        ent_coef_loss = -(ent_coef * (policy_output.logprob + target_entropy).detach()).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()
        with torch.no_grad():
            loss_info = {}
            loss_info['[ent_coef]target_ent'] = target_entropy
            loss_info['[ent_coef]ent_coef'] = ent_coef.item()
            loss_info['[ent_coef]loss'] = ent_coef_loss.item()
        return loss_info

    def get_entropy_coef(self):
        return torch.clip(self.log_ent_coef.detach().exp(), min=self.min_entropy_coef)

    def compute_critic_loss(self, experience_replay: ReplayBufferSamples, target_q_value):
        current_q_values = self.critic(experience_replay.observations, experience_replay.actions)
        check_shape(target_q_value, [self.batch_size, 1])
        critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_value) for current_q in current_q_values])
        with torch.no_grad():
            loss_info = {}
            loss_info['[critic]loss'] = critic_loss.item()
        return critic_loss, loss_info

    def compute_next_q_value(self, next_observations, next_policy_output: PolicyOuput, next_priors_output: PriorOuput,
                             next_weight_output: WeightPolicyOuput):
        next_q_values = self.critic_target(next_observations, next_policy_output.actions)
        next_q_values = torch.cat(next_q_values, dim=1)
        next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
        check_shape(next_q_values, [self.batch_size, 1])
        return next_q_values

    def compute_policy_loss(self, experience_replay: ReplayBufferSamples, policy_output: PolicyOuput,
                            priors_output: PriorOuput, weight_output: WeightPolicyOuput):
        alpha = self.get_alpha().detach()

        current_q_values = self.critic(experience_replay.observations, policy_output.actions)
        current_q_values = torch.cat(current_q_values, dim=1)
        current_q_values, _ = torch.min(current_q_values, dim=1, keepdim=True)
        with torch.no_grad():
            check_kl_divs = compute_prior_divergence(policy_output.distribution,
                                                     priors_output.distributions,
                                                     n_samples=self.n_samples,
                                                     theta=self.theta,
                                                     analytic=self.analytic_kl)
            check_nlls = compute_prior_nll(policy_output.actions, priors_output.distributions)

        kl_divs_penalty, _, kl_divs_penalty_info, kl_divs = self.compute_kl_divs_penalty(
            alpha, policy_output, weight_output, priors_output, weight_output.weights.detach())

        policy_loss = -current_q_values + kl_divs_penalty

        if self.include_entropy:
            policy_loss += self.get_entropy_coef() * policy_output.logprob[:, None]

        policy_loss = policy_loss.mean()

        with torch.no_grad():
            loss_info = {**kl_divs_penalty_info}
            loss_info['policy/kl_divs_penalty'] = kl_divs_penalty.mean().item()
            loss_info['policy/kl_divs_penalty max'] = kl_divs_penalty.max().item()
            loss_info['policy/kl_divs_penalty min'] = kl_divs_penalty.min().item()
            loss_info['policy/current_q_values'] = current_q_values.mean().item()
            loss_info['policy/current_q_values max'] = current_q_values.max().item()
            loss_info['policy/current_q_values min'] = current_q_values.min().item()
            loss_info['policy/entropy_loss'] = -(self.get_entropy_coef() * policy_output.logprob[:, None]).mean().item()
            loss_info['policy/entropy'] = -policy_output.logprob.mean().item()

            for i in range(self.n_priors):
                loss_info['policy/realized_kl_{0}'.format(i)] = check_kl_divs[:, i].mean()
                loss_info['policy/realized_nll_{0}'.format(i)] = check_nlls[:, i].mean()
                loss_info['policy/weight_{0}'.format(i)] = weight_output.weights[:, i].mean().item()
        return policy_loss, loss_info

    def update_alpha(self, priors_output: PriorOuput, weight_output: WeightPolicyOuput, policy_output: PolicyOuput):
        alpha = torch.exp(self.log_alpha)
        with torch.no_grad():
            if self.raw_kl:
                kl_divs, _, _, rkl = self.compute_kl_divs_penalty(1,
                                                                  policy_output,
                                                                  weight_output,
                                                                  priors_output,
                                                                  weight=None)
                kl_divs = kl_divs.squeeze(-1)
            else:
                with torch.no_grad():
                    min_kl_divs_distribution_output = self.weight_decoder(weight_output.weights.detach(),
                                                                          priors_output.param)
                if self.analytic_kl:
                    kl_divs = torch.distributions.kl_divergence(policy_output.distribution,
                                                                min_kl_divs_distribution_output.distribution)
                else:
                    kl_divs = mc_kl_divergence(policy_output.distribution,
                                               min_kl_divs_distribution_output.distribution,
                                               n_samples=self.n_samples,
                                               theta=self.theta)
            check_shape(kl_divs, [self.batch_size])

        loss = (alpha * (self.target_kl - kl_divs).detach()).mean()

        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            loss_info = {
                '[alpha]alpha': alpha,
                '[alpha]target': self.target_kl,
                '[alpha]kl_divs_without_alpha': (kl_divs).mean().item(),
                '[alpha]slack': (self.target_kl - kl_divs).mean().item()
            }
            if self.raw_kl:
                for i in range(self.n_priors):
                    loss_info['[alpha]realized_kl_{0}'.format(i)] = rkl[:, i].mean()
                    loss_info['[alpha]realized_nll_{0}'.format(i)] = rkl[:, i].mean()

        return loss_info

    def update_weight_actor(self, experience_replay: ReplayBufferSamples, priors_output: PriorOuput,
                            weight_output: WeightPolicyOuput):
        # Estimate Q(s,w)
        current_q_values, _ = self.estimate_weight_critic(experience_replay, priors_output, weight_output, None, None,
                                                          weight_output.weights)
        uniform = torch.ones(self.batch_size, self.n_priors, device=self.device) / self.n_priors
        lamda_annealing = self.weight_policy.lamda_annealing(self.updates)
        regularization = F.mse_loss(weight_output.weights, uniform, reduction='none').mean(axis=1, keepdim=True)
        lamda_regularization = self.lamda * lamda_annealing * regularization
        check_shape(lamda_regularization, [self.batch_size, 1])
        loss = (-current_q_values + lamda_regularization).mean()
        self.weight_actor_optimizer.zero_grad()
        loss.backward()
        self.weight_actor_optimizer.step()

        loss_info = {}
        with torch.no_grad():
            # loss_info['[weight actor]precsion'] = weight_output.precision.mean().item()
            loss_info['weight_actor/loss'] = loss.item()
            loss_info['weight_actor/current_w_q_value'] = current_q_values.mean().item()
            loss_info['weight_actor/lamda_annealing'] = lamda_annealing
            loss_info['weight_actor/regularization'] = regularization.mean().item()
            loss_info['weight_actor/lamda regularization'] = lamda_regularization.mean().item()

            if self.weight_network == 'dirichlet':
                for i in range(self.n_priors):
                    loss_info['weight_actor/mu_{0} mean'.format(i)] = weight_output.mu[:, i].mean().item()
                    loss_info['weight_actor/mu_{0} min'.format(i)] = weight_output.mu[:, i].min().item()
                    loss_info['weight_actor/mu_{0} max'.format(i)] = weight_output.mu[:, i].max().item()
            else:
                for i in range(self.n_priors):
                    loss_info['weight_actor/weight_{0} mean'.format(i)] = weight_output.weights[:, i].mean().item()
                    loss_info['weight_actor/weight_{0} min'.format(i)] = weight_output.weights[:, i].min().item()
                    loss_info['weight_actor/weight_{0} max'.format(i)] = weight_output.weights[:, i].max().item()

        return loss_info

    def estimate_weight_critic(self, experience_replay: ReplayBufferSamples, priors_output: PriorOuput,
                               weight_output: WeightPolicyOuput, next_weight_output: WeightPolicyOuput,
                               policy_output: PolicyOuput, weight):
        values = []
        # sample weight in given weight distribution
        sample_weight = weight
        sample_action_output = self.weight_decoder(sample_weight, priors_output.param)
        for _ in range(self.n_samples):
            #Q(s,a)
            sample_distribution = sample_action_output.distribution
            sample_action = sample_distribution.rsample()
            sample_action = self.max_action_range * torch.tanh(sample_action)
            sample_q_values = self.critic(experience_replay.observations, sample_action)
            sample_q_values = torch.cat(sample_q_values, dim=1)
            sample_q_values, _ = torch.min(sample_q_values, dim=1, keepdim=True)
            values.append(sample_q_values)

        values = torch.cat(values, dim=1)
        check_shape(values, [self.batch_size, self.n_samples])
        values = values.mean(dim=1, keepdim=True)
        check_shape(values, [self.batch_size, 1])

        with torch.no_grad():
            info = {}

        return values, info

    def update_weight_decoder(self, experience_replay: ReplayBufferSamples, priors_output: PriorOuput):
        weights = torch.softmax(torch.rand(priors_output.param.shape[0], self.n_priors, device=self.device) / 0.3,
                                dim=-1)

        check_shape(weights, [self.batch_size, self.n_priors])
        decoder_output = self.weight_decoder(weights, priors_output.param)
        kl_divs = compute_prior_divergence(decoder_output.distribution,
                                           priors_output.distributions,
                                           n_samples=10,
                                           theta=self.theta,
                                           analytic=self.analytic_kl)
        check_shape(kl_divs, [self.batch_size, self.n_priors])
        weigted_kl = self.compute_weighted_kl_divs(weights, kl_divs)
        loss = weigted_kl.mean()

        # update
        self.weight_decoder_optimizer.zero_grad()
        loss.backward()
        self.weight_decoder_optimizer.step()

        with torch.no_grad():
            loss_info = {'[decoder]weighted_kl': weigted_kl.mean().item()}

        return loss_info

    def compute_weighted_kl_divs(self, weight, kl_divs):
        return (weight * kl_divs).sum(dim=-1)

    def compute_kl_divs_penalty(self,
                                alpha,
                                policy_output: PolicyOuput,
                                weight_output: WeightPolicyOuput,
                                priors_output: PriorOuput,
                                weight: torch.tensor = None,
                                raw_kl=False):
        if self.raw_kl or raw_kl:
            if self.analytic_kl:
                kl_divs = [
                    torch.distributions.kl_divergence(policy_output.distribution, priors_output.distributions[i])
                    for i in range(self.n_priors)
                ]
            else:
                kl_divs = [
                    mc_kl_divergence(policy_output.distribution, priors_output.distributions[i], n_samples=10)
                    for i in range(self.n_priors)
                ]
            kl_divs = torch.stack(kl_divs, dim=1)
            kl_divs = torch.clamp(kl_divs, min=0, max=100)
            check_shape(kl_divs, [self.batch_size, self.n_priors])
            if weight is not None:
                weighted_kl_divs = self.compute_weighted_kl_divs(weight, kl_divs)[:, None]
            else:
                weight = weight_output.weights
                weighted_kl_divs = self.compute_weighted_kl_divs(weight, kl_divs)[:, None]

            check_shape(weighted_kl_divs, [self.batch_size, 1])

        else:
            with torch.no_grad():
                min_kl_divs_distribution_output = self.weight_decoder(weight_output.weights.detach(),
                                                                      priors_output.param)
            if self.analytic_kl:
                weighted_kl_divs = torch.distributions.kl_divergence(policy_output.distribution,
                                                                     min_kl_divs_distribution_output.distribution)[:,
                                                                                                                   None]
            else:
                weighted_kl_divs = mc_kl_divergence(policy_output.distribution,
                                                    min_kl_divs_distribution_output.distribution,
                                                    n_samples=self.n_samples,
                                                    theta=self.theta)[:, None]
            check_shape(weighted_kl_divs, [self.batch_size, 1])

            weighted_kl_divs = torch.clamp(weighted_kl_divs, min=0, max=100)
            kl_divs = None

        kl_divs_penalty = alpha * weighted_kl_divs
        kl_divs_penalty_info = {}

        uncertainty_discount = torch.ones(self.batch_size, 1)
        check_shape(kl_divs_penalty, [self.batch_size, 1])
        return kl_divs_penalty, uncertainty_discount, kl_divs_penalty_info, kl_divs

    def get_alpha(self):
        if self.alpha_schedule_fn:
            return torch.clamp(torch.exp(self.log_alpha), max=self.alpha_max_scheduler())
        else:
            return torch.clamp(torch.exp(self.log_alpha), min=self.min_alpha_coef)

    def _run_weight_policy(self, observations) -> WeightPolicyOuput:

        return self.weight_actors(observations)

    def _run_eval_weight_policy(self, observations) -> WeightPolicyOuput:
        with no_batchnorm_update(self.weight_actors):
            return self.weight_actors(observations)

    def _run_policy(self, observations) -> PolicyOuput:
        return self.policy(observations)

    def get_priors(self, observations) -> PriorOuput:
        return self.policy.get_prior(observations)

    def _preprocess(self, experience_replay):
        self.normalizer.update(experience_replay.next_observations)
        if self.dict_observation:
            experience_replay.observations = dict_to_torch(self.normalizer(experience_replay.observations),
                                                           expand=False,
                                                           device=self.device)
            experience_replay.next_observations = dict_to_torch(self.normalizer(experience_replay.next_observations),
                                                                expand=False,
                                                                device=self.device)
        else:
            experience_replay.observations = to_torch(self.normalizer(experience_replay.observations),
                                                      expand=False,
                                                      device=self.device)
            experience_replay.next_observations = to_torch(self.normalizer(experience_replay.next_observations),
                                                           expand=False,
                                                           device=self.device)

        experience_replay.actions = to_torch(experience_replay.actions, expand=False, device=self.device)
        experience_replay.rewards = to_torch(experience_replay.rewards, expand=False, device=self.device)
        experience_replay.dones = to_torch(experience_replay.dones, expand=False, device=self.device)

        return experience_replay

    def sample_action(self, observation) -> np.array:
        with torch.no_grad():
            observation = self.normalizer(observation)
            if self.dict_observation:
                observation = dict_to_torch(observation, expand=True, device=self.device)
            else:
                observation = to_torch(observation, expand=True, device=self.device)

            exploration_prob = 0.05
            if self.prior_exploration and self.updates < self.prior_exploration_steps:
                execute_prior_index = self.execute_prior_index
                return self.policy.execute_prior_skill(observation,
                                                       execute_prior_index).actions.squeeze(0).cpu().detach().numpy()

            elif self.policy_reuse and np.random.rand(
            ) <= exploration_prob and self.updates >= self.prior_exploration_steps:
                with no_batchnorm_update(self.weight_actors):
                    weight_output = self._run_weight_policy(observation)

                if self.updates < self.eps:
                    weight_prob = np.ones(self.n_priors) / self.n_priors
                else:
                    weight_prob = weight_output.weights.squeeze().detach().cpu().numpy()
                execute_prior_index = np.random.choice(self.n_priors, p=weight_prob)
                return self.policy.execute_prior_skill(observation,
                                                       execute_prior_index).actions.squeeze(0).cpu().detach().numpy()
            else:
                with no_batchnorm_update(self.policy):
                    return self.policy(observation).actions.squeeze(0).cpu().detach().numpy()

    def learn(self, total_timesteps: int, log_interval: int = 4):
        start_time = time.time()
        episode_reward_logger = []
        episode_reward = 0
        episode_length = 0
        num_episode = 0
        self.execute_prior_index = 0

        observation = self.env.reset()
        self.updates = 0
        while self.num_timesteps < total_timesteps:
            action = self.sample_action(observation)
            next_observation, reward, done, info = self.env.step(action)
            if 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
                self.replay_buffer.add(observation, next_observation, action, reward, False)
            else:
                self.replay_buffer.add(observation, next_observation, action, reward, done)
            observation = next_observation
            episode_reward += reward
            episode_length += 1
            self.num_timesteps += 1

            if done:
                episode_reward_logger.append(episode_reward)
                observation, episode_length, episode_reward = self.env.reset(), 0, 0
                num_episode += 1

                self.execute_prior_index = (self.execute_prior_index + 1) % self.n_priors

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:

                training_info = self.update(batch_size=self.batch_size, gradient_steps=self.gradient_steps)
                self.updates += 1

                if self.log and self.updates % log_interval == 0:
                    with torch.no_grad():
                        wandb.log({
                            **training_info, 'epispde_mean_reward': np.mean(episode_reward_logger[-100:]),
                            'num_timesteps': self.num_timesteps,
                            "num_episode": num_episode,
                            'step_ps': self.num_timesteps / (time.time() - start_time)
                        })

            if self.updates % self.checkpoint_frequency == 0:
                torch.save(
                    self.policy,
                    os.path.join(self.save_dir,
                                 "actor_checkpoint{0}.pt".format(self.updates // self.checkpoint_frequency)))
                torch.save(
                    self.critic,
                    os.path.join(self.save_dir,
                                 "critic_checkpoint{0}.pt".format(self.updates // self.checkpoint_frequency)))
                torch.save(
                    self.weight_policy,
                    os.path.join(self.save_dir,
                                 "weight_checkpoint{0}.pt".format(self.updates // self.checkpoint_frequency)))

                pickle.dump(
                    self.normalizer,
                    open(os.path.join(self.save_dir, 'N{0}.p'.format(self.updates // self.checkpoint_frequency)), "wb"))
        return self
