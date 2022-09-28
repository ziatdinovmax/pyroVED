from typing import Type, Dict, Union, List

from copy import deepcopy as dc

import torch
import torch.nn as nn

from pyro.distributions.util import broadcast_shape


def average_weights(ensemble: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Averages weights of all models in the ensemble

    Args:
        ensemble:
            Dictionary with trained weights (model's state_dict)
            of models with exact same architecture.

    Returns:
        Averaged weights (as model's state_dict)
    """
    ensemble = {k - min(ensemble.keys()): v for (k, v) in ensemble.items()}
    ensemble_state_dict = dc(ensemble[0])
    names = [name for name in ensemble_state_dict.keys() if
             name.split('_')[-1] not in ["mean", "var", "tracked"]]
    for name in names:
        w_aver = []
        for model in ensemble.values():
            for n, p in model.items():
                if n == name:
                    w_aver.append(dc(p))
        ensemble_state_dict[name].copy_(sum(w_aver) / float(len(w_aver)))
    return ensemble_state_dict


def to_onehot(idx: torch.Tensor, n: int) -> torch.Tensor:
    """
    One-hot encoding of a label
    """
    if torch.max(idx).item() >= n:
        raise AssertionError(
            "Labelling must start from 0 and "
            "maximum label value must be less than total number of classes")
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n)
    return onehot.scatter_(1, idx, 1)


class Concat(nn.Module):
    """
    Module for concatenation of tensors
    """
    def __init__(self, allow_broadcast: bool = True):
        """
        Initializes module
        """
        self.allow_broadcast = allow_broadcast
        super().__init__()

    def forward(self, input_args: Union[List[torch.Tensor], torch.Tensor]
                ) -> torch.Tensor:
        """
        Performs concatenation
        """
        if torch.is_tensor(input_args):
            return input_args
        input_args = [a.flatten(1) if a.ndim >= 4 else a for a in input_args]
        if self.allow_broadcast:
            shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
            input_args = [s.expand(shape) for s in input_args]
        out = torch.cat(input_args, dim=-1)
        return out


def _to_device(input_data: Union[torch.Tensor, List[torch.Tensor]],
               **kwargs) -> List[torch.Tensor]:
    device = kwargs.get(
            "device", 'cuda' if torch.cuda.is_available() else 'cpu')
    if len(input_data) == 1:
        input_data = input_data[0].to(device)
        return input_data
    return [t.to(device) for t in input_data]


def set_deterministic_mode(seed: int) -> None:
    """Sets all torch manual seeds.

    Parameters
    ----------
    seed : {int}
    """

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_bnorm(dim: int) -> Type[nn.Module]:
    bn_dict = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
    return bn_dict[dim]


def get_conv(dim: int) -> Type[nn.Module]:
    conv_dict = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    return conv_dict[dim]


def get_maxpool(dim: int) -> Type[nn.Module]:
    conv_dict = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
    return conv_dict[dim]


def get_activation(activation: int) -> Type[nn.Module]:
    if activation is None:
        return
    activations = {"lrelu": nn.LeakyReLU, "tanh": nn.Tanh,
                   "softplus": nn.Softplus, "relu": nn.ReLU,
                   "gelu": nn.GELU}
    return activations[activation]
