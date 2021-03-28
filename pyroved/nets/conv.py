"""
conv.py
=========

Convolutional NN modules and custom blocks

Created by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""
from typing import Union, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as tt

from ..utils import get_activation, get_bnorm, get_conv, get_maxpool

from warnings import warn, filterwarnings

filterwarnings("ignore", module="torch.nn.functional")


class convEncoderNet(nn.Module):
    """
    Standard convolutional encoder
    """
    def __init__(self,
                 input_dim: Tuple[int],
                 input_channels: int = 1,
                 latent_dim: int = 2,
                 layers_per_block: List[int] = None,
                 hidden_dim: int = 32,
                 batchnorm: bool = True,
                 activation: str = "lrelu",
                 softplus_out: bool = True,
                 pool: bool = True,
                 ) -> None:
        """
        Initializes encoder module
        """
        super(convEncoderNet, self).__init__()
        if layers_per_block is None:
            layers_per_block = [1, 2, 2]
        output_dim = (tt(input_dim) // 2**len(layers_per_block)).tolist()
        output_channels = hidden_dim * len(layers_per_block)
        self.latent_dim = latent_dim
        self.feature_extractor = FeatureExtractor(
            len(input_dim), input_channels, layers_per_block, hidden_dim,
            batchnorm, activation, pool)
        self.features2latent = features_to_latent(
            [output_channels, *output_dim], 2*latent_dim)
        self.activation_out = nn.Softplus() if softplus_out else lambda x: x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass
        """
        x = self.feature_extractor(x)
        encoded = self.features2latent(x)
        mu, sigma = encoded.split(self.latent_dim, 1)
        sigma = self.activation_out(sigma)
        return mu, sigma


class convDecoderNet(nn.Module):
    """
    Standard convolutional decoder
    """
    def __init__(self,
                 latent_dim: int,
                 output_dim: int,
                 output_channels: int = 1,
                 layers_per_block: List[int] = None,
                 hidden_dim: int = 96,
                 batchnorm: bool = True,
                 activation: str = "lrelu",
                 sigmoid_out: bool = True,
                 upsampling_mode: str = "bilinear",
                 ) -> None:
        """
        Initializes decoder module
        """
        super(convDecoderNet, self).__init__()
        if layers_per_block is None:
            layers_per_block = [2, 2, 1]
        input_dim = (tt(output_dim) // 2**len(layers_per_block)).tolist()
        self.latent2features = latent_to_features(
            latent_dim, [hidden_dim, *input_dim])
        self.upsampler = Upsampler(
            len(output_dim), hidden_dim, layers_per_block, output_channels,
            batchnorm, activation, upsampling_mode)
        self.activation_out = nn.Sigmoid() if sigmoid_out else lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        x = self.latent2features(x)
        x = self.activation_out(self.upsampler(x))
        return x


class ConvBlock(nn.Module):
    """
    Creates a block of layers each consisting of convolution operation,
    (optional) nonlinear activation and (optional) batch normalization
    """
    def __init__(self,
                 ndim: int,
                 nlayers: int,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[Tuple[int], int] = 3,
                 stride: Union[Tuple[int], int] = 1,
                 padding: Union[Tuple[int], int] = 1,
                 batchnorm: bool = False,
                 activation: str = "lrelu",
                 pool: bool = False,
                 ) -> None:
        """
        Initializes module parameters
        """
        super(ConvBlock, self).__init__()
        if not 0 < ndim < 4:
            raise AssertionError("ndim must be equal to 1, 2 or 3")
        activation = get_activation(activation)
        block = []
        for i in range(nlayers):
            input_channels = output_channels if i > 0 else input_channels
            block.append(get_conv(ndim)(input_channels, output_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding))
            if activation is not None:
                block.append(activation())
            if batchnorm:
                block.append(get_bnorm(ndim)(output_channels))
        if pool:
            block.append(get_maxpool(ndim)(2, 2))
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass
        """
        output = self.block(x)
        return output


class UpsampleBlock(nn.Module):
    """
    Upsampling performed using bilinear or nearest-neigbor interpolation
    followed by 1-by-1 convolution, which an be used to reduce a number of
    feature channels
    """
    def __init__(self,
                 ndim: int,
                 input_channels: int,
                 output_channels: int,
                 scale_factor: int = 2,
                 mode: str = "bilinear") -> None:
        """
        Initializes module parameters
        """
        super(UpsampleBlock, self).__init__()
        warn_msg = ("'bilinear' mode is not supported for 1D and 3D;" +
                    " switching to 'nearest' mode")
        if mode not in ("bilinear", "nearest"):
            raise NotImplementedError(
                "Use 'bilinear' or 'nearest' for upsampling mode")
        if not 0 < ndim < 4:
            raise AssertionError("ndim must be equal to 1, 2 or 3")
        if mode == "bilinear" and ndim in (3, 1):
            warn(warn_msg, category=UserWarning)
            mode = "nearest"
        self.mode = mode
        self.scale_factor = scale_factor
        self.conv = get_conv(ndim)(
            input_channels, output_channels,
            kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass
        """
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


class FeatureExtractor(nn.Sequential):
    """
    Convolutional feature extractor
    """
    def __init__(self,
                 ndim: int,
                 input_channels: int = 1,
                 layers_per_block: List[int] = None,
                 nfilters: int = 32,
                 batchnorm: bool = True,
                 activation: str = "lrelu",
                 pool: bool = True,
                 ) -> None:
        """
        Initializes feature extractor module
        """
        super(FeatureExtractor, self).__init__()
        if layers_per_block is None:
            layers_per_block = [1, 2, 2]
        for i, layers in enumerate(layers_per_block):
            in_filters = input_channels if i == 0 else nfilters * i
            block = ConvBlock(ndim, layers, in_filters, nfilters * (i+1),
                              batchnorm=batchnorm, activation=activation,
                              pool=pool)
            self.add_module("c{}".format(i), block)


class Upsampler(nn.Sequential):
    """
    Convolutional upsampler
    """
    def __init__(self,
                 ndim: int,
                 input_channels: int = 96,
                 layers_per_block: List[int] = None,
                 output_channels: int = 1,
                 batchnorm: bool = True,
                 activation: str = "lrelu",
                 upsampling_mode: str = "bilinear",
                 ) -> None:
        """
        Initializes upsampler module
        """
        super(Upsampler, self).__init__()
        if layers_per_block is None:
            layers_per_block = [2, 2, 1]

        nfilters = input_channels
        for i, layers in enumerate(layers_per_block):
            in_filters = nfilters if i == 0 else nfilters // i
            block = ConvBlock(ndim, layers, in_filters, nfilters // (i+1),
                              batchnorm=batchnorm, activation=activation,
                              pool=False)
            self.add_module("conv_block_{}".format(i), block)
            up = UpsampleBlock(ndim, nfilters // (i+1), nfilters // (i+1),
                               mode=upsampling_mode)
            self.add_module("up_{}".format(i), up)

        out = ConvBlock(ndim, 1, nfilters // (i+1), output_channels,
                        1, 1, 0, activation=None)
        self.add_module("output_layer", out)


class features_to_latent(nn.Module):
    """
    Maps features (usually, from a convolutional net/layer) to latent space
    """
    def __init__(self, input_dim: Tuple[int], latent_dim: int = 2) -> None:
        super(features_to_latent, self).__init__()
        self.reshape_ = torch.prod(tt(input_dim))
        self.fc_latent = nn.Linear(self.reshape_, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.reshape_)
        return self.fc_latent(x)


class latent_to_features(nn.Module):
    """
    Maps latent vector to feature space
    """
    def __init__(self, latent_dim: int, out_dim: Tuple[int]) -> None:
        super(latent_to_features, self).__init__()
        self.reshape_ = out_dim
        self.fc = nn.Linear(latent_dim, torch.prod(tt(out_dim)).item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x.view(-1, *self.reshape_)
