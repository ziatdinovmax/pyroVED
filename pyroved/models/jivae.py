"""
jivae.py
=========

Variational autoencoder for learning (jointly) discrete and
continuous latent representations of data with arbitrary affine transformations
(rotations, translations, and scale)

Created by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""
from typing import Tuple, Union, List

import pyro
import pyro.distributions as dist
import torch

from ..nets import fcDecoderNet, jfcEncoderNet, sDecoderNet
from ..utils import (generate_grid, generate_latent_grid,
                     generate_latent_grid_traversal, get_sampler,
                     plot_grid_traversal, plot_img_grid, plot_spect_grid,
                     set_deterministic_mode, to_onehot, transform_coordinates)
from .base import baseVAE

tt = torch.tensor


class jiVAE(baseVAE):
    """
    Variational autoencoder for learning (jointly) discrete and
    continuous latent representations of data while enforcing rotational,
    translational, and scale invariances.

    Args:
        data_dim:
            Dimensionality of the input data; (h x w) for images
            or (length,) for spectra.
        latent_dim:
            Number of continuous latent dimensions.
        discrete_dim:
            Number of discrete latent dimensions.
        invariances:
            List with invariances to enforce. For 2D systems, `r` enforces
            rotational invariance, `t` enforces invariance to
            translations, `sc` enforces a scale invariance, and
            invariances=None corresponds to vanilla VAE.
            For 1D systems, 't' enforces translational invariance and
            invariances=None is vanilla VAE
        hidden_dim_e:
            List with the number of hidden units in each layer of
            encoder (inference network). Defaults to [128, 128].
        hidden_dim_d:
            List with the number of hidden units in each layer of
            decoder (generator network). Defau;ts to [128, 128].
        activation:
            Non-linear activation for inner layers of encoder and decoder.
            The available activations are ReLU ('relu'), leaky ReLU ('lrelu'),
            hyberbolic tangent ('tanh'), softplus ('softplus'), and GELU ('gelu').
            (The default is 'tanh').
        sampler_d:
            Decoder sampler, as defined as p(x|z) = sampler(decoder(z)).
            The available samplers are 'bernoulli', 'continuous_bernoulli',
            and 'gaussian' (Default: 'bernoulli').
        sigmoid_d:
            Sigmoid activation for the decoder output (Default: True).
        seed:
            Seed used in torch.manual_seed(seed) and
            torch.cuda.manual_seed_all(seed).

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

    Initialize and train a joint VAE model with rotational invariance for 10 discrete classes

    >>> import torch
    >>> import pyroved as pv
    >>> # initialize joint rVAE model
    >>> data_dim = (28, 28)
    >>> jrvae = jiVAE(data_dim, latent_dim=2, discrete_dim=10, invariances=['r'])
    >>> # Initialize trainer (in pyroVED, we use parallel enumeration instead of Gumbel-Softmax approximation)
    >>> trainer = pv.trainers.SVItrainer(jvae, lr=1e-3, enumerate_parallel=True)
    >>> # Use "time"-dependent KL scale factor for continuous latent variables
    >>> kl_scale = torch.cat(  
    >>>    [torch.ones(10,) * 40,  # put pressure on the continuous latent channel at the beginning
    >>>     torch.linspace(40, 3, 40)]  # gradually release the pressure
    >>> ) 
    >>> # Train the model
    >>> for e in range(200):
    >>>     sc = kl_scale[e] if e < len(kl_scale) else kl_scale[-1]
    >>>     trainer.step(train_loader, scale_factor=[sc, 3])  # [continuous, discrete] KL scale factors
    >>>     trainer.print_statistics()
    >>>     # Plot the traversal of the latent manifold learned so far
    >>>     if (e + 1) % 10 == 0:
    >>>         for i in range(2):
    >>>             jvae.manifold_traversal(10, i, cmap='viridis'); 
    """

    def __init__(self,
                 data_dim: Tuple[int],
                 latent_dim: int,
                 discrete_dim: int,
                 invariances: List[str] = None,
                 hidden_dim_e: List[int] = None,
                 hidden_dim_d: List[int] = None,
                 activation: str = "tanh",
                 sampler_d: str = "bernoulli",
                 sigmoid_d: bool = True,
                 seed: int = 1,
                 **kwargs: Union[str, float]
                 ) -> None:
        """
        Initializes j-iVAE's modules and parameters
        """
        args = (data_dim, invariances)
        super(jiVAE, self).__init__(*args, **kwargs)
        pyro.clear_param_store()
        set_deterministic_mode(seed)
        self.data_dim = data_dim

        # Initialize the Encoder NN
        self.encoder_z = jfcEncoderNet(
            data_dim, latent_dim+self.coord, discrete_dim,
            hidden_dim_e, activation, softplus_out=True)

        # Initialize the Decoder NN
        dnet = sDecoderNet if 0 < self.coord < 5 else fcDecoderNet
        self.decoder = dnet(
            data_dim, latent_dim, discrete_dim, hidden_dim_d,
            activation, sigmoid_out=sigmoid_d, unflat=False)

        # Initialize the decoder's sampler
        self.sampler_d = get_sampler(sampler_d, **kwargs)

        # Set continuous and discrete dimensions
        self.z_dim = latent_dim + self.coord
        self.discrete_dim = discrete_dim

        # Move model parameters to appropriate device
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
                phi, dx, sc, z = self.split_latent(z.repeat(self.discrete_dim, 1))
                if 't' in self.invariances:
                    dx = (dx * self.t_prior).unsqueeze(1)
                # transform coordinate grid
                grid = self.grid.expand(bdim*self.discrete_dim, *self.grid.shape)
                x_coord_prime = transform_coordinates(grid, phi, dx, sc)
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

    def split_latent(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Split latent variable into parts with coordinates transformations
        and image content
        """
        return self._split_latent(z)

    def encode(self,
               x_new: torch.Tensor,
               logits: bool = False,
               **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network.
        The output is a tuple with mean and standard deviations of
        encoded distributions and a predicted class.

        Args:
            x_new:
                Data to encode with a trained j-iVAE. The new data must have
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
        Decodes a batch of latent coordinates using a trained generator (decoder) network.

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
                    and plot parameters ('padding', 'pad_value', 'cmap', 'origin', 'ylim')
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
                and plot parameters ('padding', 'pad_value', 'cmap', 'origin', 'ylim')
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
