"""
fc.py

Module for creating fully-connected encoder and decoder modules

Created by Maxim Ziatdinov (ziatdinovmax@gmail.com)
"""

from typing import List, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.tensor as tt
from pyro.distributions.util import broadcast_shape

from ..utils import get_activation


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
        if self.allow_broadcast:
            shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
            input_args = [s.expand(shape) for s in input_args]
        out = torch.cat(input_args, dim=-1)
        return out


class fcEncoderNet(nn.Module):
    """
    Standard fully-connected encoder NN for VAE.
    The encoder outputs mean and standard evidation of the encoded distribution.
    """
    def __init__(self,
                 in_dim: Tuple[int],
                 latent_dim: int = 2,
                 num_classes: int = 0,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 activation: str = 'tanh',
                 softplus_out: bool = True,
                 flat: bool = True
                 ) -> None:
        """
        Initializes module
        """
        super(fcEncoderNet, self).__init__()
        if len(in_dim) not in [1, 2, 3]:
            raise ValueError("in_dim must be (h, w), (h, w, c), or (l,)")
        self.in_dim = torch.prod(tt(in_dim)).item() + num_classes
        self.flat = flat

        self.concat = Concat()
        self.fc_layers = make_fc_layers(
            self.in_dim, hidden_dim, num_layers, activation)
        self.fc11 = nn.Linear(hidden_dim, latent_dim)
        self.fc12 = nn.Linear(hidden_dim, latent_dim)
        self.activation_out = nn.Softplus() if softplus_out else lambda x: x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass
        """
        x = self.concat(x)
        if self.flat:
            x = x.view(-1, self.in_dim)
        x = self.fc_layers(x)
        mu = self.fc11(x)
        sigma = self.activation_out(self.fc12(x))
        return mu, sigma


class jfcEncoderNet(nn.Module):
    """
    Fully-connected encoder for joint VAE.
    The encoder outputs mean, standard evidation and class probabilities.
    """
    def __init__(self,
                 in_dim: Tuple[int],
                 latent_dim: int = 2,
                 discrete_dim: int = 0,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 activation: str = 'tanh',
                 softplus_out: bool = True,
                 flat: bool = True
                 ) -> None:
        """
        Initializes module
        """
        super(jfcEncoderNet, self).__init__()
        if len(in_dim) not in [1, 2, 3]:
            raise ValueError("in_dim must be (h, w), (h, w, c), or (l,)")
        self.in_dim = torch.prod(tt(in_dim)).item()
        self.flat = flat

        self.concat = Concat()
        self.fc_layers = make_fc_layers(
            self.in_dim, hidden_dim, num_layers, activation)
        self.fc11 = nn.Linear(hidden_dim, latent_dim)
        self.fc12 = nn.Linear(hidden_dim, latent_dim)
        self.fc13 = nn.Linear(hidden_dim, discrete_dim)
        self.activation_out = nn.Softplus() if softplus_out else lambda x: x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass
        """
        x = self.concat(x)
        if self.flat:
            x = x.view(-1, self.in_dim)
        x = self.fc_layers(x)
        mu = self.fc11(x)
        sigma = self.activation_out(self.fc12(x))
        alpha = torch.softmax(self.fc13(x), dim=-1)
        return mu, sigma, alpha


class fcDecoderNet(nn.Module):
    """
    Standard fully-connected decoder for VAE
    """
    def __init__(self,
                 out_dim: Tuple[int],
                 latent_dim: int,
                 num_classes: int = 0,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 activation: str = 'tanh',
                 sigmoid_out: bool = True,
                 unflat: bool = True
                 ) -> None:
        """
        Initializes module
        """
        super(fcDecoderNet, self).__init__()
        if len(out_dim) not in [1, 2, 3]:
            raise ValueError("in_dim must be (h, w), (h, w, c), or (l,)")
        self.unflat = unflat
        if self.unflat:
            self.reshape = out_dim
        out_dim = torch.prod(tt(out_dim)).item()

        self.concat = Concat()
        self.fc_layers = make_fc_layers(
            latent_dim+num_classes, hidden_dim, num_layers, activation)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.activation_out = nn.Sigmoid() if sigmoid_out else lambda x: x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        z = self.concat(z)
        x = self.fc_layers(z)
        x = self.activation_out(self.out(x))
        if self.unflat:
            return x.view(-1, *self.reshape)
        return x


class sDecoderNet(nn.Module):
    """
    Spatial generator (decoder) network with fully-connected layers
    """
    def __init__(self,
                 out_dim: Tuple[int],
                 latent_dim: int,
                 num_classes: int = 0,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 activation: str = 'tanh',
                 sigmoid_out: bool = True,
                 unflat: bool = True
                 ) -> None:
        """
        Initializes module
        """
        super(sDecoderNet, self).__init__()
        if len(out_dim) not in [1, 2, 3]:
            raise ValueError("in_dim must be (h, w), (h, w, c), or (l,)")
        self.unflat = unflat
        if self.unflat:
            self.reshape = out_dim
        coord_dim = 1 if len(out_dim) < 2 else 2

        self.concat = Concat()
        self.coord_latent = coord_latent(
            latent_dim+num_classes, hidden_dim, coord_dim)
        self.fc_layers = make_fc_layers(
            hidden_dim, hidden_dim, num_layers, activation)
        self.out = nn.Linear(hidden_dim, 1)  # need to generalize to multi-channel (c > 1)
        self.activation_out = nn.Sigmoid() if sigmoid_out else lambda x: x

    def forward(self, x_coord: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        z = self.concat(z)
        x = self.coord_latent(x_coord, z)
        x = self.fc_layers(x)
        x = self.activation_out(self.out(x))
        if self.unflat:
            return x.view(-1, *self.reshape)
        return x


class coord_latent(nn.Module):
    """
    The "spatial" part of the trVAE's decoder that allows for translational
    and rotational invariance (based on https://arxiv.org/abs/1909.11663)
    """
    def __init__(self,
                 latent_dim: int,
                 out_dim: int,
                 ndim: int = 2,
                 activation_out: bool = True) -> None:
        """
        Initializes module
        """
        super(coord_latent, self).__init__()
        self.fc_coord = nn.Linear(ndim, out_dim)
        self.fc_latent = nn.Linear(latent_dim, out_dim, bias=False)
        self.activation = nn.Tanh() if activation_out else None

    def forward(self,
                x_coord: torch.Tensor,
                z: Tuple[torch.Tensor]) -> torch.Tensor:
        batch_dim, n = x_coord.size()[:2]
        x_coord = x_coord.reshape(batch_dim * n, -1)
        h_x = self.fc_coord(x_coord)
        h_x = h_x.reshape(batch_dim, n, -1)
        h_z = self.fc_latent(z)

        h_z = h_z.view(-1, h_z.size(-1))
        h = h_x.add(h_z.unsqueeze(1))
        h = h.reshape(batch_dim * n, -1)
        if self.activation is not None:
            h = self.activation(h)
        return h


class fcClassifierNet(nn.Module):
    """
    Simple classification neural network with fully-connected layers only.
    """
    def __init__(self,
                 in_dim: Tuple[int],
                 num_classes: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 activation: str = 'tanh'
                 ) -> None:
        """
        Initializes module
        """
        super(fcClassifierNet, self).__init__()
        if len(in_dim) not in [1, 2, 3]:
            raise ValueError("in_dim must be (h, w), (h, w, c), or (l,)")
        self.in_dim = torch.prod(tt(in_dim)).item()

        self.fc_layers = make_fc_layers(
            self.in_dim, hidden_dim, num_layers, activation)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        x = self.fc_layers(x)
        x = self.out(x)
        return torch.softmax(x, dim=-1)


def make_fc_layers(in_dim: int,
                   hidden_dim: int = 128,
                   num_layers: int = 2,
                   activation: str = "tanh"
                   ) -> Type[nn.Module]:
    """
    Generates a module with stacked fully-connected (aka dense) layers
    """
    fc_layers = []
    for i in range(num_layers):
        hidden_dim_ = in_dim if i == 0 else hidden_dim
        fc_layers.extend(
            [nn.Linear(hidden_dim_, hidden_dim),
             get_activation(activation)()])
    fc_layers = nn.Sequential(*fc_layers)
    return fc_layers
