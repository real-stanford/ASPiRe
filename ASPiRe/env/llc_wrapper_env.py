from collections import deque
from logging import info
import gym
from gym import Env
from gym.spaces import Box
import torch
import numpy as np

from ASPiRe.rl.utility.helper import check_shape, get_image_obs


class llc_wrapper_env(Env):
    """[summary]
    The env action space is [-1,1]
    The wrapped_env action space is [-1,1]
    """

    def __init__(self, wrapped_env: Env, H_dim, action_dim, skills_decoder, info_fn=None, device=0) -> None:
        super().__init__()
        self.device = device
        self._wrapped_env = wrapped_env
        self.H_dim = H_dim
        self.action_space = Box(low=-1, high=1, shape=(action_dim,))
        self.observation_space = self._wrapped_env.observation_space
        self._wrapped_env_action_space = wrapped_env.action_space

        self.skills_decoder = skills_decoder.eval().to(self.device)
        self.dict_observation = True if type(self.observation_space) == gym.spaces.dict.Dict else False
        # HARD CODE:
        self.dict_observation = False
        self.info_fn = info_fn

    def decode(self, z: torch.Tensor):
        with torch.no_grad():
            action_sequence = self.skills_decoder(
                z.repeat(1, self.H_dim).reshape(-1, self.H_dim, self.action_space.shape[0]))
            action_sequence = action_sequence.squeeze(0).cpu().numpy()
            action_sequence = np.clip(action_sequence, self._wrapped_env_action_space.low,
                                      self._wrapped_env_action_space.high)
            check_shape(action_sequence, [self.H_dim, self._wrapped_env_action_space.shape[0]])
            return action_sequence

    def reset(self):
        obs = self._wrapped_env.reset()
        if self.dict_observation:
            self.image_queue = deque(maxlen=2)
            self.image_queue.append(obs['image'])
            obs = {'vector': obs['vector'], 'image': get_image_obs(self.image_queue, 2)}
        else:
            obs = self._wrapped_env.reset()
        return obs

    def step(self, action: np.array):
        action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_sequence = self.decode(action)
        # Bacuse we assume wrapped env act space is [-1,1]
        action_sequence = np.clip(action_sequence, -1, 1)

        seq_reward = 0
        seq_info = []
        for i in range(self.H_dim):
            next_env_observation, reward, done, info = self._wrapped_env.step(action_sequence[i, :])
            seq_info.append(info)
            if self.dict_observation:
                self.image_queue.append(next_env_observation['image'])
            seq_reward = seq_reward + reward
            if done:
                break

        reward = seq_reward
        obs = self._get_obs(next_env_observation)
        if self.info_fn:
            info = self.info_fn(seq_info)
        else:
            info = {}
        return obs, reward, done, info

    def _get_obs(self, next_env_observation):
        if self.dict_observation:
            return {'vector': next_env_observation['vector'], 'image': get_image_obs(self.image_queue, 2)}
        else:
            return next_env_observation
