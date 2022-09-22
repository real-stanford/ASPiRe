from typing import List
import numpy as np
from contextlib import contextmanager
from torch.nn.modules import BatchNorm1d, BatchNorm2d, BatchNorm3d
import torch
import random


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


class ParamDict(AttrDict):

    def overwrite(self, new_params):
        for param in new_params:
            # print('overriding param {} to value {}'.format(param, new_params[param]))
            self.__setattr__(param, new_params[param])
        return self


def to_torch(data, expand=False, device=0):
    if expand:
        return torch.tensor(data, device=device, dtype=torch.float32).unsqueeze(0)
    return torch.tensor(data, device=device, dtype=torch.float32)


def to_numpy(data):
    return data.detach().cpu().numpy()


def dict_to_torch(dict, expand, device):
    ret = {}
    for k in list(dict.keys()):
        if expand:
            ret[k] = torch.tensor(dict[k], dtype=torch.float32, device=device).unsqueeze(0)
        else:
            ret[k] = torch.tensor(dict[k], dtype=torch.float32, device=device)

    return ret


def get_image_obs(image_queue, past_frame):
    queue_len = len(image_queue)
    if queue_len > 0:
        return np.concatenate([image_queue[max(-i - 1, -queue_len)] for i in reversed(range(past_frame))], axis=0)
    else:
        return None


def check_shape(t, target_shape):
    if not list(t.shape) == target_shape:
        raise ValueError(f"Temsor should have shape {target_shape} but has shape {list(t.shape)}!")


def check_gradient(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item()**2
    total_norm = total_norm**0.5

    return total_norm


def switch_off_batchnorm_update(model):
    """Switches off batchnorm update in all submodules of model."""
    for module in model.modules():
        if isinstance(module, BatchNorm1d) \
                or isinstance(module, BatchNorm2d) \
                or isinstance(module, BatchNorm3d):
            module.eval()


def switch_on_batchnorm_update(model):
    """Switches on batchnorm update in all submodules of model."""
    for module in model.modules():
        if isinstance(module, BatchNorm1d) \
                or isinstance(module, BatchNorm2d) \
                or isinstance(module, BatchNorm3d):
            module.train()


@contextmanager
def no_batchnorm_update(model):
    """Switches off all batchnorm updates within context."""
    switch_off_batchnorm_update(model)
    yield
    switch_on_batchnorm_update(model)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)