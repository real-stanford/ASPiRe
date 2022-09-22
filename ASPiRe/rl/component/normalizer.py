from typing import List
import numpy as np

### copy from stablebaseline


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(self.mean, self.var, self.count,
                                                                             batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class Normalizer():

    def __init__(self, shape=(), clip_raw_obs=np.array(float("Inf")), clip_norm_obs=np.array(float("Inf"))):
        self._running_mean_std = RunningMeanStd(shape=shape)
        self.clip_raw_obs = clip_raw_obs
        self.clip_norm_obs = clip_norm_obs

    def reset(self, shape=()):
        self._running_mean_std = RunningMeanStd(shape=shape)

    def update(self, vals):
        if isinstance(vals, list):
            vals = np.stack(vals)
        self._running_mean_std.update(vals)

    def __call__(self, vals):
        """Performs normalization."""
        vals = self._clip(vals, range=self.clip_raw_obs)
        return self._clip((vals - self._running_mean_std.mean) / np.sqrt(self._running_mean_std.var),
                          range=self.clip_norm_obs)

    @staticmethod
    def _clip(val, range):
        return np.clip(val, -range, range)

    @property
    def mean(self):
        return self._running_mean_std.mean

    @property
    def std(self):
        return np.sqrt(self._running_mean_std.var)


class Dict_Normalizer():

    def __init__(self, entry_info: List = []):
        self.entry_normalizer = {}
        self.entry = []
        for e in entry_info:
            self.entry_normalizer[e[0]] = Normalizer(e[1])
            self.entry.append(e[0])

    def update(self, vals):
        for key in self.entry:
            self.entry_normalizer[key].update(vals[key])

    def __call__(self, vals):
        return_vals = {}
        for key in vals:
            if key in self.entry:
                return_vals[key] = self.entry_normalizer[key](vals[key])
            else:
                return_vals[key] = vals[key]
        return return_vals


class Dict_Constant_Normalizer():

    def __init__(self, entry_info: List = [], constant=1, offset=None):
        self.entry_normalizer = {}
        self.entry = []
        self.constant = constant
        self.offset = offset
        for e in entry_info:
            self.entry_normalizer[e[0]] = Normalizer(e[1])
            self.entry.append(e[0])

    def update(self, vals):
        pass

    def __call__(self, vals):
        return_vals = {}
        for key in vals:
            if key in self.entry:
                if self.offset is None:
                    return_vals[key] = vals[key] / self.constant
                else:
                    target_vals = vals[key]
                    if target_vals.ndim == 2:
                        return_vals[key] = np.concatenate(
                            [target_vals[:, :self.offset], target_vals[:, self.offset:] / self.constant], axis=1)
                    else:
                        return_vals[key] = np.concatenate(
                            [target_vals[:self.offset], target_vals[self.offset:] / self.constant], axis=0)
            else:
                return_vals[key] = vals[key]
        return return_vals


class Dummay_Normalizer(Normalizer):

    def __init__(self, shape=(), clip_raw_obs=np.array(float("Inf")), clip_norm_obs=np.array(float("Inf"))):
        super().__init__(shape=shape, clip_raw_obs=clip_raw_obs, clip_norm_obs=clip_norm_obs)

    def update(self, vals):
        pass

    def __call__(self, vals):
        return vals