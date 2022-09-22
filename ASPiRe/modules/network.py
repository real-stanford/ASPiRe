import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from ASPiRe.modules.distributions import MultivariateDiagonalNormal
from typing import Dict, Iterable, Optional, Tuple, Union, List, Type, NamedTuple


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    use_batch_norm: bool = False,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        if use_batch_norm:
            modules.append(nn.BatchNorm1d(num_features=net_arch[idx]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


STD_MAX = 8.0
STD_MIN = 1e-3

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


class skill_encoder(nn.Module):

    def __init__(self, H_dim, action_dim, latent_dim=10, lstm_hidden=128, enc_size=32):
        super().__init__()
        self.embed = nn.Linear(action_dim, lstm_hidden)
        self.lstm = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.out = nn.Linear(lstm_hidden, enc_size)
        self.fc_mu = nn.Linear(enc_size, latent_dim)
        self.fc_log_std = nn.Linear(enc_size, latent_dim)
        self.lstm_hidden = lstm_hidden

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [(batch, seq_len, input_size):]
            output: BXlatent_dim
        """
        batch_size = x.shape[0]
        input_size = x.shape[-1]
        x = x.reshape(-1, input_size)  # [batch*seq_len,input_size]
        embed = self.embed(x)  #[batch*seq_len,lstm_hidden]
        embed = embed.reshape(batch_size, -1, self.lstm_hidden)  #[batch,seq_len,lstm_hidden]
        o, _ = self.lstm(embed)  #[batch,seq_len,lstm_hidden]
        o = o[:, -1, :]  # take last output cell: [bxlstm_hidden]
        o = F.relu(self.out(o))  #[batch,enc_size]
        mu = self.fc_mu(o)  #[batch,latent_dim]
        log_std = self.fc_log_std(o)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return MultivariateDiagonalNormal(mu, log_std.exp())


class skill_decoder(nn.Module):

    def __init__(self, H_dim, action_dim, input_dim=10, lstm_hidden=128, dec_size=32):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.action_dim = action_dim
        self.H_dim = H_dim

        self.embed = nn.Linear(input_dim, lstm_hidden)
        # input_dim=latent_dim
        self.lstm = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.out = nn.Linear(lstm_hidden, dec_size)

        self.action_layer = nn.Linear(dec_size, action_dim)

    def forward(self, x):
        #x:(batch x seq_len x input_size/dim)
        batch_size = x.shape[0]
        input_size = x.shape[-1]
        x = x.reshape(-1, input_size)  # [batch*seq_len,input_size]
        embed = self.embed(x)  #[batch*seq_len,lstm_hidden]
        embed = embed.reshape(batch_size, -1, self.lstm_hidden)  #[batch,seq_len,lstm_hidden]
        o, _ = self.lstm(embed)  #[batch,seq_len,lstm_hidden]
        o = o.reshape(-1, self.lstm_hidden)  #[batch* H_dim,lstm_hidden ]
        o = F.relu(self.out(o))  #[batch*H_dim,dec_size]
        action = self.action_layer(o)
        action = action.reshape(batch_size, self.H_dim, self.action_dim)

        return action


class SkillPrior(nn.Module):

    def __init__(
        self,
        H_dim,
        action_dim,
        latent_dim,
        hidden_dim,
        kl_coef,
        beta,
        nll,
        priors_num,
        device,
    ):
        super().__init__()

        self.skill_encoder = skill_encoder(H_dim, action_dim, latent_dim, hidden_dim)
        self.skill_decoder = skill_decoder(H_dim, action_dim, latent_dim, hidden_dim)
        self.priors_num = priors_num
        self.skill_priors = nn.ModuleList()

        self.kl_coef = kl_coef
        self.beta = beta
        self.H_dim = H_dim
        self.latent_dim = latent_dim
        self.nll = nll
        self.device = device

    def add(self, prior):

        self.skill_priors.append(prior)

    def forward(self, state, action, prior_index):
        p = self.skill_priors[prior_index](state)
        q = self.skill_encoder(action)

        z = q.rsample()

        action_hat = self.skill_decoder(z.repeat(1, self.H_dim).reshape(-1, self.H_dim, self.latent_dim))
        rec_loss = -MultivariateDiagonalNormal(loc=action_hat, scale_diag=1).log_prob(action).sum(axis=-1)
        kl_loss = self.kl_coef * torch.distributions.kl.kl_divergence(q.distribution.base_dist,
                                                                      self.fixed_prior).sum(axis=-1)

        if self.nll:
            prior_loss = -self.beta * p.log_prob(z.clone().detach())
        else:
            q_detach = MultivariateDiagonalNormal(q.mean.clone().detach(), q.stddev.clone().detach())
            prior_loss = self.beta * torch.distributions.kl.kl_divergence(q_detach, p)

        output = AttrDict()
        output.rec_loss = rec_loss
        output.kl_loss = kl_loss
        output.prior_loss = prior_loss
        output.loss = rec_loss + kl_loss + prior_loss
        output.action = action
        return output

    @property
    def fixed_prior(self):
        return Normal(0, 1)


class skill(nn.Module):

    def __init__(self, net_arch=[128] * 7, act_limit=1, action_dim=10, preprocessor=None, device=None):
        super().__init__()
        self.preprocessor = preprocessor
        input_dim = self.preprocessor.output_dim
        latent_policy = create_mlp(input_dim=input_dim, output_dim=-1, net_arch=net_arch, use_batch_norm=True)
        self.latent_policy = nn.Sequential(*latent_policy)
        self.mu = nn.Linear(net_arch[-1], action_dim)
        self.log_std = nn.Linear(net_arch[-1], action_dim)

    def forward(self, observation):
        feature = self.preprocessor(observation)
        policy_latent = self.latent_policy(feature)
        mu = self.mu(policy_latent)
        log_std = self.log_std(policy_latent)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return MultivariateDiagonalNormal(mu, log_std.exp())
