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

from ..utils import get_activation, get_bnorm, get_conv, get_maxpool

from warnings import warn, filterwarnings

filterwarnings("ignore", module="torch.nn.functional")

tt = torch.tensor


class convEncoderNet(nn.Module):
    """
    Standard convolutional encoder
    """
    def __init__(self,
                 input_dim: Tuple[int],
                 latent_dim: int = 2,
                 input_channels: int = 1,
                 hidden_dim: List[int] = None,
                 batchnorm: bool = False,
                 activation: str = "lrelu",
                 softplus_out: bool = True,
                 pool_last: bool = False,
                 ) -> None:
        """
        Initializes encoder module
        """
        super(convEncoderNet, self).__init__()
        if hidden_dim is None:
            hidden_dim = [(32,), (64, 64), (128, 128)]
        dim_denom = 2**len(hidden_dim) if pool_last else 2**(len(hidden_dim) - 1)
        output_dim = torch.div(tt(input_dim), dim_denom).int().tolist()
        output_channels = hidden_dim[-1][-1]
        self.latent_dim = latent_dim
        self.feature_extractor = FeatureExtractor(
            len(input_dim), input_channels, hidden_dim,
            batchnorm=batchnorm, activation=activation,
            pool_last=pool_last)
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
                 hidden_dim: List[int] = None,
                 batchnorm: bool = False,
                 activation: str = "lrelu",
                 sigmoid_out: bool = True,
                 upsampling_mode: str = "bilinear",
                 ) -> None:
        """
        Initializes decoder module
        """
        super(convDecoderNet, self).__init__()
        if hidden_dim is None:
            hidden_dim = [(128, 128), (64, 64), (32,)]
        input_dim = torch.div(tt(output_dim), 2**len(hidden_dim)).int().tolist()
        self.latent2features = latent_to_features(
            latent_dim, [hidden_dim[0][0], *input_dim])
        self.upsampler = Upsampler(
            len(output_dim), hidden_dim[0][0], hidden_dim, output_channels,
            batchnorm=batchnorm, activation=activation,
            upsampling_mode=upsampling_mode)
        self.activation_out = nn.Sigmoid() if sigmoid_out else lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        x = self.latent2features(x)
        x = self.activation_out(self.upsampler(x))
        return x


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
                 conv_filters: List[int] = None,
                 kernel_size: Union[Tuple[int], int] = 3,
                 stride: Union[Tuple[int], int] = 1,
                 padding: Union[Tuple[int], int] = 1,
                 batchnorm: bool = False,
                 activation: str = "lrelu",
                 pool_last: bool = True,
                 ) -> None:
        """
        Initializes feature extractor module
        """
        super(FeatureExtractor, self).__init__()
        if not 0 < ndim < 4:
            raise AssertionError("ndim must be equal to 1, 2 or 3")
        if conv_filters is None:
            conv_filters = [(32,), (64, 64), (128, 128)]
        activation = get_activation(activation)
        all_layers = []
        j = 0
        for i, cblock in enumerate(conv_filters):
            for ch in cblock:
                if i == j == 0:
                    ch_in = input_channels
                else:
                    for l in all_layers[::-1]:
                        if hasattr(l, "out_channels"):
                            ch_in = l.out_channels
                            break
                all_layers.append(
                    get_conv(ndim)(ch_in, ch, kernel_size, stride, padding))  
                if activation is not None:
                    all_layers.append(activation())
                if batchnorm:
                    all_layers.append(get_bnorm(ndim)(ch))
                j += 1
            if j + 1 < sum([len(c) for c in conv_filters]):
                all_layers.append(get_maxpool(ndim)(2, 2))
            else:
                if pool_last:
                    all_layers.append(get_maxpool(ndim)(2, 2))
        self.layers = nn.Sequential(*all_layers)
        
    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Upsampler(nn.Sequential):
    """
    Convolutional upsampler
    """
    def __init__(self,
                 ndim: int,
                 input_channels: int = 128,
                 conv_filters: List[int] = None,
                 output_channels: int = 1,
                 kernel_size: Union[Tuple[int], int] = 3,
                 stride: Union[Tuple[int], int] = 1,
                 padding: Union[Tuple[int], int] = 1,
                 batchnorm: bool = False,
                 activation: str = "lrelu",
                 upsampling_mode: str = "bilinear",
                 ) -> None:
        """
        Initializes upsampler module
        """
        super(Upsampler, self).__init__()
        if not 0 < ndim < 4:
            raise AssertionError("ndim must be equal to 1, 2 or 3")
        if conv_filters is None:
            conv_filters = [(128, 128), (64, 64), (32,)]
        activation = get_activation(activation)
        all_layers = []
        j = 0
        for i, cblock in enumerate(conv_filters):
            for ch in cblock:
                if i == j == 0:
                    ch_in = input_channels
                else:
                    for l in all_layers[::-1]:
                        if hasattr(l, "out_channels"):
                            ch_in = l.out_channels
                            break
                all_layers.append(
                    get_conv(ndim)(ch_in, ch, kernel_size, stride, padding))  
                if activation is not None:
                    all_layers.append(activation())
                if batchnorm:
                    all_layers.append(get_bnorm(ndim)(ch))
                j += 1
            all_layers.append(
                UpsampleBlock(ndim, ch, ch, mode=upsampling_mode))
        all_layers.append(
            get_conv(ndim)(ch, output_channels, 1, 1, 0))
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
            

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
