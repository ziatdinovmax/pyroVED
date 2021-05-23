from typing import Type

import torch
import torch.nn as nn


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
