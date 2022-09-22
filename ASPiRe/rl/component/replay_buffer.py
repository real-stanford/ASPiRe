from typing import Any, Dict, List, Optional, Tuple, Type, Union, NamedTuple

import numpy as np
import gym
from CompositeSkill.rl.utility.helper import AttrDict


class ReplayBufferSamples(NamedTuple):
    observations: np.array
    next_observations: np.array
    actions: np.array
    rewards: np.array
    dones: np.array


class ReplayBuffer:

    def __init__(self, size, env: gym.Env):
        self.obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.act_buf = np.zeros((size, *env.action_space.shape), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, next_obs, act, rew, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = AttrDict(observations=self.obs_buf[idxs],
                         next_observations=self.next_obs_buf[idxs],
                         actions=self.act_buf[idxs],
                         rewards=self.rew_buf[idxs],
                         dones=self.done_buf[idxs])
        return batch


class DictReplayBuffer:

    def __init__(self, size, action_dim, obs_entry_info):
        self.obs_buf = {}
        self.next_obs_buf = {}
        for e in obs_entry_info:
            self.obs_buf[e[0]] = np.zeros((size, *e[1]), dtype=np.float32)
        for e in obs_entry_info:
            self.next_obs_buf[e[0]] = np.zeros((size, *e[1]), dtype=np.float32)

        self.act_buf = np.zeros((size, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, next_obs, act, rew, done):
        for k in list(self.obs_buf.keys()):
            self.obs_buf[k][self.ptr] = obs[k]

        for k in list(self.next_obs_buf.keys()):
            self.next_obs_buf[k][self.ptr] = next_obs[k]

        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        observations = {k: self.obs_buf[k][idxs] for k in self.obs_buf.keys()}
        next_observations = {k: self.next_obs_buf[k][idxs] for k in self.next_obs_buf.keys()}
        batch = AttrDict(observations=observations,
                         next_observations=next_observations,
                         actions=self.act_buf[idxs],
                         rewards=self.rew_buf[idxs],
                         dones=self.done_buf[idxs])
        return batch
