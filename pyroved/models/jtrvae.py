"""
jtrvae.py
=========

Variational autoencoder for learning (jointly) discrete and
continuous latent representations on data with arbitrary rotations
and/or translations

Created by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""
from typing import Tuple, Union

import pyro
import pyro.distributions as dist
import torch
import torch.tensor as tt

from ..nets import fcDecoderNet, jfcEncoderNet, sDecoderNet
from ..utils import (generate_grid, generate_latent_grid,
                     generate_latent_grid_traversal, get_sampler,
                     plot_grid_traversal, plot_img_grid, plot_spect_grid,
                     set_deterministic_mode, to_onehot, transform_coordinates)
from .base import baseVAE


class jtrVAE(baseVAE):
    """
    Variational autoencoder for learning (jointly) discrete and
    continuous latent representations on data with arbitrary rotations
    and/or translations

    Args:
        data_dim:
            Dimensionality of the input data; (h x w) for images
            or (length,) for spectra.
        latent_dim:
            Number of continuous latent dimensions.
        discrete_dim:
            Number of discrete latent dimensions.
        coord:
            For 2D systems, *coord=0* is vanilla VAE, *coord=1* enforces
            rotational invariance, *coord=2* enforces invariance to translations,
            and *coord=3* enforces both rotational and translational invariances.
            For 1D systems, *coord=0* is vanilla VAE and *coord>0* enforces
            translational invariance.
        hidden_dim_e:
            Number of hidden units per each layer in encoder (inference network).
        hidden_dim_d:
            Number of hidden units per each layer in decoder (generator network).
        num_layers_e:
            Number of layers in encoder (inference network).
        num_layers_d:
            Number of layers in decoder (generator network).
        activation:
            Non-linear activation for inner layers of encoder and decoder.
            The available activations are ReLU ('relu'), leaky ReLU ('lrelu'),
            hyberbolic tangent ('tanh'), and softplus ('softplus').
            The default activation is 'tanh'.
        sampler_d:
            Decoder sampler, as defined as p(x|z) = sampler(decoder(z)).
            The available samplers are 'bernoulli', 'continuous_bernoulli',
            and 'gaussian' (Default: 'bernoulli').
        sigmoid_d:
            Sigmoid activation for the decoder output (Default: True).
        seed:
            Seed used in torch.manual_seed(seed) and
            torch.cuda.manual_seed_all(seed).
        kwargs:
            Additional keyword arguments are *dx_prior* and *dy_prior* for setting
            a translational prior(s), and *decoder_sig* for setting sigma
            in the decoder's sampler when it is set to "gaussian".

    Example:

    Initialize a joint VAE model with rotational invariance for 10 discrete classes

    >>> data_dim = (28, 28)
    >>> jrvae = jtrVAE(data_dim, latent_dim=2, discrete_dim=10, coord=1)
    """

    def __init__(self,
                 data_dim: Tuple[int],
                 latent_dim: int,
                 discrete_dim: int,
                 coord: int = 0,
                 hidden_dim_e: int = 128,
                 hidden_dim_d: int = 128,
                 num_layers_e: int = 2,
                 num_layers_d: int = 2,
                 activation: str = "tanh",
                 sampler_d: str = "bernoulli",
                 sigmoid_d: bool = True,
                 seed: int = 1,
                 **kwargs: float
                 ) -> None:
        """
        Initializes jtrVAE's modules and parameters
        """
        super(jtrVAE, self).__init__()
        pyro.clear_param_store()
        set_deterministic_mode(seed)
        self.ndim = len(data_dim)
        self.data_dim = data_dim
        if self.ndim == 1 and coord > 0:
            coord = 1
        self.encoder_z = jfcEncoderNet(
            data_dim, latent_dim+coord, discrete_dim, hidden_dim_e,
            num_layers_e, activation, softplus_out=True)
        if coord not in [0, 1, 2, 3]:
            raise ValueError("'coord' argument must be 0, 1, 2 or 3")
        dnet = sDecoderNet if coord in [1, 2, 3] else fcDecoderNet
        self.decoder = dnet(
            data_dim, latent_dim, discrete_dim, hidden_dim_d,
            num_layers_d, activation, sigmoid_out=sigmoid_d, unflat=False)
        self.sampler_d = get_sampler(sampler_d, **kwargs)
        self.z_dim = latent_dim + coord
        self.coord = coord
        self.discrete_dim = discrete_dim
        self.grid = generate_grid(data_dim).to(self.device)
        dx_pri = tt(kwargs.get("dx_prior", 0.1))
        dy_pri = kwargs.get("dy_prior", dx_pri.clone())
        t_prior = tt([dx_pri, dy_pri]) if self.ndim == 2 else dx_pri
        self.t_prior = t_prior.to(self.device)
        self.to(self.device)

    def model(self,
              x: torch.Tensor,
              **kwargs: float) -> None:
        """
        Defines the model p(x|z,c)p(z)p(c)
        """
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        # KLD scale factor (see e.g. https://openreview.net/pdf?id=Sy2fzU9gl)
        beta = kwargs.get("scale_factor", [1., 1.])
        if isinstance(beta, (float, int, list)):
            beta = torch.tensor(beta)
        if beta.ndim == 0:
            beta = torch.tensor([beta, beta])
        reshape_ = torch.prod(tt(x.shape[1:])).item()
        bdim = x.shape[0]
        with pyro.plate("data"):
            # sample the continuous latent vector from the constant prior distribution
            z_loc = x.new_zeros(torch.Size((bdim, self.z_dim)))
            z_scale = x.new_ones(torch.Size((bdim, self.z_dim)))
            # sample discrete latent vector from the constant prior
            alpha = x.new_ones(torch.Size((bdim, self.discrete_dim))) / self.discrete_dim
            # sample from prior (value will be sampled by guide when computing ELBO)
            with pyro.poutine.scale(scale=beta[0]):
                z = pyro.sample("latent_cont", dist.Normal(z_loc, z_scale).to_event(1))
            with pyro.poutine.scale(scale=beta[1]):
                z_disc = pyro.sample("latent_disc", dist.OneHotCategorical(alpha))
            # split latent variable into parts for rotation and/or translation
            # and image content
            if self.coord > 0:
                phi, dx, z = self.split_latent(z.repeat(self.discrete_dim, 1))
                if torch.sum(dx.abs()) != 0:
                    dx = (dx * self.t_prior).unsqueeze(1)
                # transform coordinate grid
                grid = self.grid.expand(bdim*self.discrete_dim, *self.grid.shape)
                x_coord_prime = transform_coordinates(grid, phi, dx)
            # Continuous and discrete latent variables for the decoder
            z = [z, z_disc.reshape(-1, self.discrete_dim) if self.coord > 0 else z_disc]
            # decode the latent code z together with the transformed coordinates (if any)
            dec_args = (x_coord_prime, z) if self.coord else (z,)
            loc = self.decoder(*dec_args)
            # score against actual images/spectra
            loc = loc.view(*z_disc.shape[:-1], reshape_)
            pyro.sample(
                "obs", self.sampler_d(loc).to_event(1),
                obs=x.view(-1, reshape_))

    def guide(self,
              x: torch.Tensor,
              **kwargs: float) -> None:
        """
        Defines the guide q(z,c|x)
        """
        # register PyTorch module `encoder_z` with Pyro
        pyro.module("encoder_z", self.encoder_z)
        # KLD scale factor (see e.g. https://openreview.net/pdf?id=Sy2fzU9gl)
        beta = kwargs.get("scale_factor", [1., 1.])
        if isinstance(beta, (float, int, list)):
            beta = torch.tensor(beta)
        if beta.ndim == 0:
            beta = torch.tensor([beta, beta])
        with pyro.plate("data"):
            # use the encoder to get the parameters used to define q(z,c|x)
            z_loc, z_scale, alpha = self.encoder_z(x)
            # sample the latent code z
            with pyro.poutine.scale(scale=beta[0]):
                pyro.sample("latent_cont", dist.Normal(z_loc, z_scale).to_event(1))
            with pyro.poutine.scale(scale=beta[1]):
                pyro.sample("latent_disc", dist.OneHotCategorical(alpha))

    def split_latent(self, zs: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Split latent variable into parts with rotation and/or translation
        and image content
        """
        if self.ndim == 1:
            dx = zs[:, 0:1]
            zs = zs[:, 1:]
            return None, dx, zs
        phi, dx = tt(0), tt(0)
        # rotation + translation
        if self.coord == 3:
            phi = zs[:, 0]  # encoded angle
            dx = zs[:, 1:3]  # translation
            zs = zs[:, 3:]  # image content
        # translation only
        elif self.coord == 2:
            dx = zs[:, :2]
            zs = zs[:, 2:]
        # rotation only
        elif self.coord == 1:
            phi = zs[:, 0]
            zs = zs[:, 1:]
        return phi, dx, zs

    def encode(self,
               x_new: torch.Tensor,
               logits: bool = False,
               **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network

        Args:
            x_new:
                Data to encode with a trained jtrVAE. The new data must have
                the same dimensions (images height and width or spectra length)
                as the one used for training.
            logits:
                Return raw class probabilities (Default: False).
            kwargs:
                Batch size as 'batch_size' (for encoding large volumes of data).
        """
        z = self._encode(x_new)
        z_loc = z[:, :self.z_dim]
        z_scale = z[:, self.z_dim:2*self.z_dim]
        classes = z[:, 2*self.z_dim:]
        if not logits:
            _, classes = torch.max(classes, 1)
        return z_loc, z_scale, classes

    def decode(self, z: torch.Tensor, y: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Decodes a batch of latent coordinates

        Args:
            z: Latent coordinates (without rotational and translational parts)
            y: Classes as one-hot vectors for each sample in z
        """
        z = torch.cat([z.to(self.device), y.to(self.device)], -1)
        loc = self._decode(z, **kwargs)
        return loc.view(-1, *self.data_dim)

    def manifold2d(self, d: int, disc_idx: int = 0, plot: bool = True,
                   **kwargs: Union[str, int, float]) -> torch.Tensor:
        """
        Plots a learned latent manifold in the data space

        Args:
            d: Grid size
            disc_idx: Discrete dimension for which we plot continuous latent manifolds
            plot: Plots the generated manifold (Default: True)
            kwargs: Keyword arguments include custom min/max values for grid
                    boundaries passed as 'z_coord' (e.g. z_coord = [-3, 3, -3, 3]),
                    'angle' and 'shift' to condition a generative model on,
                    and plot parameters ('padding', 'padding_value', 'cmap', 'origin', 'ylim')
        """
        z, (grid_x, grid_y) = generate_latent_grid(d, **kwargs)
        z_disc = to_onehot(tt(disc_idx).unsqueeze(0), self.discrete_dim)
        z_disc = z_disc.repeat(z.shape[0], 1)
        loc = self.decode(z, z_disc, **kwargs)
        if plot:
            if self.ndim == 2:
                plot_img_grid(
                    loc, d,
                    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                    **kwargs)
            elif self.ndim == 1:
                plot_spect_grid(loc, d, **kwargs)
        return loc

    def manifold_traversal(self, d: int, cont_idx: int, cont_idx_fixed: int = 0,
                           plot: bool = True, **kwargs: Union[str, int, float]
                           ) -> torch.Tensor:
        """
        Latent space traversal for joint continuous and discrete
        latent representations

        Args:
            d: Grid size
            cont_idx:
                Continuous latent variable used for plotting
                a latent manifold traversal
            cont_idx_fixed:
                Value which the remaining continuous latent variables are fixed at
            plot:
                Plots the generated manifold (Default: True)
            kwargs:
                Keyword arguments include custom min/max values for grid
                boundaries passed as 'z_coord' (e.g. z_coord = [-3, 3, -3, 3]),
                'angle' and 'shift' to condition a generative model one,
                and plot parameters ('padding', 'padding_value', 'cmap', 'origin', 'ylim')
        """
        num_samples = d**2
        disc_dim = self.discrete_dim
        cont_dim = self.z_dim - self.coord
        data_dim = self.data_dim
        # Get continuous and discrete latent coordinates
        samples_cont, samples_disc = generate_latent_grid_traversal(
            d, cont_dim, disc_dim, cont_idx, cont_idx_fixed, num_samples)
        # Pass discrete and continuous latent coordinates through a decoder
        decoded = self.decode(samples_cont, samples_disc, **kwargs)
        if plot:
            plot_grid_traversal(decoded, d, data_dim, disc_dim, **kwargs)
        return decoded
