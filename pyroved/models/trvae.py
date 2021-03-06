"""
trvae.py
=========

Variational autoencoder with rotational and/or translational invariances

Created by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""

from typing import Optional, Tuple, Union

import pyro
import pyro.distributions as dist
import torch

from pyroved.models.base import baseVAE
from pyroved.nets import fcDecoderNet, fcEncoderNet, sDecoderNet
from pyroved.utils import (
    generate_grid, generate_latent_grid, get_sampler,
    plot_img_grid, plot_spect_grid, set_deterministic_mode,
    to_onehot, transform_coordinates
)


class trVAE(baseVAE):
    """Variational autoencoder that enforces rotational and/or translational
    invariances..

    Args:
        data_dim:
            Dimensionality of the input data; use (height x width) for images
            or (length,) for spectra.
        latent_dim:
            Number of latent dimensions.
        coord:
            For 2D systems, `coord=0` is vanilla VAE, `coord=1` enforces
            rotational invariance, `coord=2` enforces invariance to
            translations, and `coord=3` enforces both rotational and
            translational invariances. For 1D systems, `coord=0` is vanilla VAE
            and `coord>0` enforces transaltional invariance. Must be 0, 1, 2 or
            3.
        num_classes:
            Number of classes (if any) for class-conditioned (t)(r)VAE (The
            default is 0).
        hidden_dim_e:
            Number of hidden units per each layer in encoder (inference
            network). (The default is 128).
        hidden_dim_d:
            Number of hidden units per each layer in decoder (generator
            network). (The default is 128).
        num_layers_e:
            Number of layers in encoder (inference network). (The default is
            2).
        num_layers_d:
            Number of layers in decoder (generator network). (The default is
            2).
        activation:
            Non-linear activation for inner layers of encoder and decoder.
            The available activations are ReLU ('relu'), leaky ReLU ('lrelu'),
            hyberbolic tangent ('tanh'), and softplus ('softplus')
            The default activation is 'tanh'. (The default is "tanh").
        sampler_d:
            Decoder sampler, as defined as p(x|z) = sampler(decoder(z)).
            The available samplers are 'bernoulli', 'continuous_bernoulli',
            and 'gaussian'. (The default is "bernoulli").
        sigmoid_d:
            Sigmoid activation for the decoder output. (The default is True).
        seed:
            Seed used in torch.manual_seed(seed) and
            torch.cuda.manual_seed_all(seed). (The default is 1).

    Keyword Args:
        device:
            Sets device to which model and data will be moved.
            Defaults to 'cuda:0' if a GPU is available and to CPU otherwise.
        dx_prior:
            Translational prior in x direction (float between 0 and 1)
        dx_prior:
            Translational prior in y direction (float between 0 and 1)
        decoder_sig:
            Sets sigma for a "gaussian" decoder sampler

    Raises:
        ValueError:
            If coord is not equal to 0, 1, 2 or 3.

    Examples:
        Initialize a VAE model with rotational invariance

        >>> data_dim = (28, 28)
        >>> rvae = trVAE(data_dim, latent_dim=2, coord=1)

        Initialize a class-conditioned VAE model with rotational invariance
        for dataset that has 10 classes

        >>> data_dim = (28, 28)
        >>> rvae = trVAE(data_dim, latent_dim=2, num_classes=10, coord=1)
    """

    def __init__(
        self,
        data_dim: Tuple[int],
        latent_dim: int = 2,
        coord: int = 3,
        num_classes: int = 0,
        hidden_dim_e: int = 128,
        hidden_dim_d: int = 128,
        num_layers_e: int = 2,
        num_layers_d: int = 2,
        activation: str = "tanh",
        sampler_d: str = "bernoulli",
        sigmoid_d: bool = True,
        seed: int = 1,
        **kwargs: Union[str, float]
         ) -> None:

        if coord not in [0, 1, 2, 3]:
            raise ValueError("`coord` argument must be 0, 1, 2 or 3")

        super(trVAE, self).__init__(**kwargs)

        self.coord = coord

        # Reset the pyro ParamStoreDict object's dictionaries.
        pyro.clear_param_store()

        set_deterministic_mode(seed)  # Set all torch manual seeds

        # Silently assign coord=1 for one-dimensional data when user-supplied
        # coord value > 0.
        self.ndim = len(data_dim)
        if self.ndim == 1 and self.coord > 0:
            self.coord = 1

        # Initialize the encoder network
        self.encoder_z = fcEncoderNet(
            data_dim, latent_dim + self.coord, 0, hidden_dim_e, num_layers_e,
            activation, softplus_out=True
        )

        # Initialize the decoder network
        dnet = sDecoderNet if self.coord in [1, 2, 3] else fcDecoderNet
        self.decoder = dnet(
            data_dim, latent_dim, num_classes, hidden_dim_d, num_layers_d,
            activation, sigmoid_out=sigmoid_d
        )

        # Initialize the decoder's sampler
        self.sampler_d = get_sampler(sampler_d, **kwargs)

        # Sets continuous and discrete dimensions
        self.z_dim = latent_dim + self.coord
        self.num_classes = num_classes

        # Generates coordinates grid
        self.grid = generate_grid(data_dim)

        # Prior "belief" about the degree of translational disorder in the system
        dx_pri = torch.tensor(kwargs.get("dx_prior", 0.1))
        dy_pri = kwargs.get("dy_prior", dx_pri.clone())
        t_prior = torch.tensor([dx_pri, dy_pri]) if self.ndim == 2 else dx_pri

        # Send objects to their appropriate devices.
        self.grid = self.grid.to(self.device)
        self.t_prior = t_prior.to(self.device)
        self.to(self.device)

    def model(self,
              x: torch.Tensor,
              y: Optional[torch.Tensor] = None,
              **kwargs: float) -> None:
        """
        Defines the model p(x|z)p(z)
        """
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        # KLD scale factor (see e.g. https://openreview.net/pdf?id=Sy2fzU9gl)
        beta = kwargs.get("scale_factor", 1.)
        reshape_ = torch.prod(torch.tensor(x.shape[1:])).item()
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            with pyro.poutine.scale(scale=beta):
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            if self.coord > 0:  # rotationally- and/or translationaly-invariant mode
                # Split latent variable into parts for rotation
                # and/or translation and image content
                phi, dx, z = self.split_latent(z)
                if torch.sum(dx.abs()) != 0:
                    dx = (dx * self.t_prior).unsqueeze(1)
                # transform coordinate grid
                grid = self.grid.expand(x.shape[0], *self.grid.shape)
                x_coord_prime = transform_coordinates(grid, phi, dx)
            # Add class label (if any)
            if y is not None:
                z = torch.cat([z, y], dim=-1)
            # decode the latent code z together with the transformed coordinates (if any)
            dec_args = (x_coord_prime, z) if self.coord else (z,)
            loc = self.decoder(*dec_args)
            # score against actual images ("binary cross-entropy loss")
            pyro.sample(
                "obs", self.sampler_d(loc.view(-1, reshape_)).to_event(1),
                obs=x.view(-1, reshape_))

    def guide(self,
              x: torch.Tensor,
              y: Optional[torch.Tensor] = None,
              **kwargs: float) -> None:
        """
        Defines the guide q(z|x)
        """
        # register PyTorch module `encoder_z` with Pyro
        pyro.module("encoder_z", self.encoder_z)
        # KLD scale factor (see e.g. https://openreview.net/pdf?id=Sy2fzU9gl)
        beta = kwargs.get("scale_factor", 1.)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder_z(x)
            # sample the latent code z
            with pyro.poutine.scale(scale=beta):
                pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def split_latent(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Split latent variable into parts for rotation
        and/or translation and image content
        """
        # For 1D, there is only a translation
        if self.ndim == 1:
            dx = z[:, 0:1]
            z = z[:, 1:]
            return None, dx, z
        phi, dx = torch.tensor(0), torch.tensor(0)
        # rotation + translation
        if self.coord == 3:
            phi = z[:, 0]  # encoded angle
            dx = z[:, 1:3]  # translation
            z = z[:, 3:]  # image content
        # translation only
        elif self.coord == 2:
            dx = z[:, :2]
            z = z[:, 2:]
        # rotation only
        elif self.coord == 1:
            phi = z[:, 0]
            z = z[:, 1:]
        return phi, dx, z

    def encode(self, x_new: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network

        Args:
            x_new:
                Data to encode with a trained trVAE. The new data must have
                the same dimensions (images height and width or spectra length)
                as the one used for training.
            kwargs:
                Batch size as 'batch_size' (for encoding large volumes of data)
        """
        z = self._encode(x_new)
        z_loc, z_scale = z.split(self.z_dim, 1)
        return z_loc, z_scale

    def decode(self,
               z: torch.Tensor,
               y: torch.Tensor = None,
               **kwargs: int) -> torch.Tensor:
        """
        Decodes a batch of latent coordnates

        Args:
            z: Latent coordinates (without rotational and translational parts)
            y: Class (if any) as a batch of one-hot vectors
            kwargs: Batch size as 'batch_size'
        """
        z = z.to(self.device)
        if y is not None:
            z = torch.cat([z, y.to(self.device)], -1)
        loc = self._decode(z, **kwargs)
        return loc

    def manifold2d(self, d: int, plot: bool = True,
                   **kwargs: Union[str, int, float]) -> torch.Tensor:
        """
        Plots a learned latent manifold in the image space

        Args:
            d: Grid size
            plot: Plots the generated manifold (Default: True)
            kwargs: Keyword arguments include 'label' for class label (if any),
                    custom min/max values for grid boundaries passed as 'z_coord'
                    (e.g. z_coord = [-3, 3, -3, 3]), 'angle' and 'shift' to condition
                    a generative model on, and plot parameters
                    ('padding', 'padding_value', 'cmap', 'origin', 'ylim')
        """
        z, (grid_x, grid_y) = generate_latent_grid(d, **kwargs)
        z = [z]
        if self.num_classes > 0:
            cls = torch.tensor(kwargs.get("label", 0))
            if cls.ndim < 2:
                cls = to_onehot(cls.unsqueeze(0), self.num_classes)
            z = z + [cls.repeat(z[0].shape[0], 1)]
        loc = self.decode(*z, **kwargs)
        if plot:
            if self.ndim == 2:
                plot_img_grid(
                    loc, d,
                    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                    **kwargs)
            elif self.ndim == 1:
                plot_spect_grid(loc, d, **kwargs)
        return loc
