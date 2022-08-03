"""
ivae.py
=======

Variational autoencoder with invariance to rotations, translations, and scale

Created by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""

from typing import Optional, Tuple, Union, List

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


class iVAE(baseVAE):
    """
    Variational autoencoder that enforces rotational, translational,
    and scale invariances.

    Args:
        data_dim:
            Dimensionality of the input data; use (height x width) for images
            or (length,) for spectra.
        latent_dim:
            Number of latent dimensions.
        invariances:
            List with invariances to enforce. For 2D systems, `r` enforces
            rotational invariance, `t` enforces invariance to
            translations, `sc` enforces a scale invariance, and
            invariances=None corresponds to vanilla VAE.
            For 1D systems, 't' enforces translational invariance and
            invariances=None is vanilla VAE
        c_dim:
            "Feature dimension" of the c vector in p(z|c) where z is
            explicitly conditioned on variable c. The latter can be continuous
            or discrete. For example, to train a class-conditional VAE on
            a dataset with 10 classes, the c_dim must be equal to 10 and
            the corresponding n x 10 vector should represent one-hot encoded labels.
            (The default c_dim value is 0, i.e. no conditioning is performed).
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
            hyberbolic tangent ('tanh'), softplus ('softplus'), and GELU ('gelu').
            (The default is 'tanh').
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
        dy_prior:
            Translational prior in y direction (float between 0 and 1)
        sc_prior:
            Scale prior (usually, sc_prior << 1)
        decoder_sig:
            Sets sigma for a "gaussian" decoder sampler

    Examples:
        Initialize a VAE model with rotational invariance

        >>> data_dim = (28, 28)
        >>> rvae = iVAE(data_dim, latent_dim=2, invariances=['r'])

        Initialize a class-conditional VAE model with rotational and
        translational invarainces for dataset that has 10 classes

        >>> data_dim = (28, 28)
        >>> rvae = iVAE(data_dim, latent_dim=2, c_dim=10, invariances=['r', 't'])
    """

    def __init__(
        self,
        data_dim: Tuple[int],
        latent_dim: int = 2,
        invariances: List[str] = None,
        c_dim: int = 0,
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
        args = (data_dim, invariances)
        super().__init__(*args, **kwargs)

        # Reset the pyro ParamStoreDict object's dictionaries
        pyro.clear_param_store()
        # Set all torch manual seeds
        set_deterministic_mode(seed)

        # Initialize the encoder network
        self.encoder_z = fcEncoderNet(
            data_dim, latent_dim + self.coord, 0, hidden_dim_e, num_layers_e,
            activation, softplus_out=True
        )
        # Initialize the decoder network
        dnet = sDecoderNet if 0 < self.coord < 5 else fcDecoderNet
        self.decoder = dnet(
            data_dim, latent_dim, c_dim, hidden_dim_d, num_layers_d,
            activation, sigmoid_out=sigmoid_d
        )
        # Initialize the decoder's sampler
        self.sampler_d = get_sampler(sampler_d, **kwargs)

        # Sets continuous and discrete dimensions
        self.z_dim = latent_dim + self.coord
        self.c_dim = c_dim

        # Move model parameters to appropriate device
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
                phi, dx, sc, z = self.split_latent(z)
                if 't' in self.invariances:
                    dx = (dx * self.t_prior).unsqueeze(1)
                # transform coordinate grid
                grid = self.grid.expand(x.shape[0], *self.grid.shape)
                x_coord_prime = transform_coordinates(grid, phi, dx, sc)
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
        return self._split_latent(z)

    def encode(self, x_new: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network

        Args:
            x_new:
                Data to encode with a trained (i)VAE model. The new data must have
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
            y: Conditional "property" vector (e.g. one-hot encoded class vector)
            kwargs: Batch size as 'batch_size'
        """
        z = z.to(self.device)
        if y is not None:
            z = torch.cat([z, y.to(self.device)], -1)
        loc = self._decode(z, **kwargs)
        return loc

    def manifold2d(
        self, d: int,
        y: torch.Tensor = None,
        plot: bool = True,
        figsize: Tuple[float] = (8.0, 8.0),
        latents: Tuple[int] = [0, 1],
        **kwargs: Union[str, int, float]
    ) -> torch.Tensor:
        """Plots a learned latent manifold in the image space

        Args:
            d:
                Grid size
            plot:
                Plots the generated manifold (Default: True)
            y:
                Conditional "property" vector (e.g. one-hot encoded class
                vector)
            kwargs:
                Keyword arguments include custom min/max values for grid
                boundaries passed as 'z_coord' (e.g. z_coord = [-3, 3, -3, 3]),
                'angle' and 'shift' to condition a generative model on, and
                plot parameters ('padding', 'padding_value', 'cmap', 'origin',
                'ylim').
        """
        z, (grid_x, grid_y) = generate_latent_grid(d, **kwargs)

        # We silence all other latent variables except those listed in the
        # latent list by setting them to zero. Note this choice is arbitrary
        # and is just a consequence of only being able to easily visualize
        # two axes at once.
        z_tmp = torch.zeros(size=(z.shape[0], self.z_dim - self.coord))
        z_tmp[:, latents[0]] = z[:, 0]
        z_tmp[:, latents[1]] = z[:, 1]
        z = [z_tmp]

        if self.c_dim > 0:
            if y is None:
                raise ValueError(
                    "To generate a manifold pass a conditional vector y"
                )
            y = y.unsqueeze(1) if 0 < y.ndim < 2 else y
            z = z + [y.expand(z[0].shape[0], *y.shape[1:])]

        loc = self.decode(*z, **kwargs)

        if plot:
            if self.ndim == 2:
                plot_img_grid(
                    loc, d, extent=[
                        grid_x.min(), grid_x.max(), grid_y.min(),
                        grid_y.max()
                    ], figsize=figsize, **kwargs)
            elif self.ndim == 1:
                plot_spect_grid(loc, d, figsize=figsize, **kwargs)
        return loc
