"""
ss_reg_ivae.py
==============

Variational autoencoder for semi-supervised regression
with an option to enforce rotational, positional and scale
invariances

Created by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""
from typing import List, Optional, Tuple, Type, Union

import pyro
import pyro.distributions as dist
import torch

from ..nets import fcDecoderNet, fcEncoderNet, fcRegressorNet, sDecoderNet
from ..utils import (generate_latent_grid, get_sampler, init_dataloader,
                     plot_img_grid, plot_spect_grid, set_deterministic_mode,
                     transform_coordinates)
from .base import baseVAE


class ss_reg_iVAE(baseVAE):
    """
    Semi-supervised variational autoencoder for regression tasks
    with the enforcement of rotational, translational, and scale invariances.

    Args:
        data_dim:
            Dimensionality of the input data; use (h x w) for images
            or (length,) for spectra.
        latent_dim:
            Number of latent dimensions.
        reg_dim:
            Number of output dimensions in regression. For example,
            for a single output regressor, specify reg_dim=1.
        invariances:
            List with invariances to enforce. For 2D systems, `r` enforces
            rotational invariance, `t` enforces invariance to
            translations, `sc` enforces a scale invariance, and
            invariances=None corresponds to vanilla VAE.
            For 1D systems, 't' enforces translational invariance and
            invariances=None is vanilla VAE
        hidden_dim_e:
            List with the number of hidden units in each layer
            of encoder (inference network). Defautls to [128, 128].
        hidden_dim_d:
            List with the number of hidden units in each layer
            of decoder (generator network). Defaults to [128, 128].
        hidden_dim_reg:
            List with the number of hidden units in each layer of regression NN.
            Defaults to [128, 128].
        activation:
            Non-linear activation for inner layers of both encoder and the decoder.
            The available activations are ReLU ('relu'), leaky ReLU ('lrelu'),
            hyberbolic tangent ('tanh'), softplus ('softplus'), and GELU ('gelu').
            (The default is "tanh").
        sampler_d:
            Decoder sampler, as defined as p(x|z) = sampler(decoder(z)).
            The available samplers are 'bernoulli', 'continuous_bernoulli',
            and 'gaussian' (Default: 'bernoulli').
        sigmoid_d:
            Sigmoid activation for the decoder output (Default: True)
        seed:
            Seed used in torch.manual_seed(seed) and
            torch.cuda.manual_seed_all(seed)

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
        regressor_sig:
            Sets sigma for a regression sampler

    Examples:

    Initialize a VAE model with rotational invariance for
    a semi-supervised single-output regression.

    >>> # Initialize ssVAE
    >>> data_dim = (28, 28)
    >>> ssvae = ss_reg_iVAE(data_dim, latent_dim=2, reg_dim=1, invariances=['r'])
    >>> # Initialize SVI trainer for regression
    >>> trainer = pv.trainers.auxSVItrainer(ssvae, task='regression')
    >>> # Get dataloaders
    >>> loader_unlabeled, loader_labeled, loader_val = pv.utils.init_ssvae_dataloaders(
    >>>     X_unlabeled, (X_labeled, y_labels), (X_val, y_val))
    >>> # Train for 100 epochs:
    >>> for e in range(100):
    >>>     trainer.step(loader_unlabeled, loader_labeled, loader_val, aux_loss_multiplier=200)
    >>>     trainer.print_statistics()
    >>>     if e > 80:  # save running weights of the regression NN at the end of training trajectory
    >>>         trainer.save_running_weights("encoder_y")
    >>> # average saved weights for the regression NN (to improve prediction accuracy)
    >>> trainer.average_weights("encoder_y")
    """
    def __init__(self,
                 data_dim: Tuple[int],
                 latent_dim: int,
                 reg_dim: int,
                 invariances: List[str] = None,
                 hidden_dim_e: List[int] = None,
                 hidden_dim_d: List[int] = None,
                 hidden_dim_reg: List[int] = None,
                 activation: str = "tanh",
                 sampler_d: str = "bernoulli",
                 sigmoid_d: bool = True,
                 seed: int = 1,
                 **kwargs: Union[str, float]
                 ) -> None:
        """
        Initializes ss_reg_iVAE parameters
        """
        args = (data_dim, invariances)
        super(ss_reg_iVAE, self).__init__(*args, **kwargs)
        pyro.clear_param_store()
        set_deterministic_mode(seed)

        self.data_dim = data_dim

        # Initialize z-Encoder neural network
        self.encoder_z = fcEncoderNet(
            data_dim, latent_dim+self.coord, reg_dim,
            hidden_dim_e, activation, flat=False)

        # Initialize y-Encoder neural network
        self.encoder_y = fcRegressorNet(
            data_dim, reg_dim, hidden_dim_reg, activation)

        # Initializes Decoder neural network
        dnet = sDecoderNet if 0 < self.coord < 5 else fcDecoderNet
        self.decoder = dnet(
            data_dim, latent_dim, reg_dim, hidden_dim_d,
            activation, sigmoid_out=sigmoid_d, unflat=False)
        self.sampler_d = get_sampler(sampler_d, **kwargs)

        # Set sigma for regression sampler
        self.reg_sig = kwargs.get("regressor_sig", 0.5)

        # Sets continuous and discrete dimensions
        self.z_dim = latent_dim + self.coord
        self.reg_dim = reg_dim

        # Send model parameters to their appropriate devices.
        self.to(self.device)

    def model(self,
              xs: torch.Tensor,
              ys: Optional[torch.Tensor] = None,
              **kwargs: float) -> None:
        """
        Model of the generative process p(x|z,y)p(y)p(z)
        """
        pyro.module("ss_vae", self)
        batch_dim = xs.size(0)
        specs = dict(dtype=xs.dtype, device=xs.device)
        beta = kwargs.get("scale_factor", 1.)
        # pyro.plate enforces independence between variables in batches xs, ys
        with pyro.plate("data"):
            # sample the latent vector from the constant prior distribution
            prior_loc = torch.zeros(batch_dim, self.z_dim, **specs)
            prior_scale = torch.ones(batch_dim, self.z_dim, **specs)
            with pyro.poutine.scale(scale=beta):
                zs = pyro.sample(
                    "z", dist.Normal(prior_loc, prior_scale).to_event(1))
            # split latent variable into parts for rotation and/or translation
            # and image content
            if self.coord > 0:
                phi, dx, sc, zs = self.split_latent(zs)
                if 't' in self.invariances:
                    dx = (dx * self.t_prior).unsqueeze(1)
                # transform coordinate grid
                grid = self.grid.expand(zs.shape[0], *self.grid.shape)
                x_coord_prime = transform_coordinates(grid, phi, dx, sc)
            # sample label from the constant prior or observe the value
            c_prior = (torch.zeros(batch_dim, self.reg_dim, **specs))
            ys = pyro.sample(
                "y", dist.Normal(c_prior, self.reg_sig).to_event(1), obs=ys)
            # Score against the parametrized distribution
            # p(x|y,z) = bernoulli(decoder(y,z))
            d_args = (x_coord_prime, [zs, ys]) if self.coord else ([zs, ys],)
            loc = self.decoder(*d_args)
            loc = loc.view(*ys.shape[:-1], -1)
            pyro.sample("x", self.sampler_d(loc).to_event(1), obs=xs.flatten(1))

    def guide(self, xs: torch.Tensor,
              ys: Optional[torch.Tensor] = None,
              **kwargs: float) -> None:
        """
        Guide q(z|y,x)q(y|x)
        """
        beta = kwargs.get("scale_factor", 1.)
        with pyro.plate("data"):
            # sample and score the digit with the variational distribution
            # q(y|x) = categorical(alpha(x))
            if ys is None:
                c = self.encoder_y(xs)
                ys = pyro.sample("y", dist.Normal(c, self.reg_sig).to_event(1))
            # sample and score the latent vector with the variational
            # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            loc, scale = self.encoder_z([xs, ys])
            with pyro.poutine.scale(scale=beta):
                pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def split_latent(self, zs: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Split latent variable into parts with rotation and/or translation
        and image content
        """
        zdims = list(zs.shape)
        zdims[-1] = zdims[-1] - self.coord
        zs = zs.view(-1, zs.size(-1))
        # For 1D, there is only translation
        phi, dx, sc, zs = self._split_latent(zs)
        return phi, dx, sc, zs.view(*zdims)

    def model_aux(self, xs: torch.Tensor,
                  ys: Optional[torch.Tensor] = None,
                  **kwargs: float) -> None:
        """
        Models an auxiliary (supervised) loss
        """
        pyro.module("ss_vae", self)
        with pyro.plate("data"):
            # the extra term to yield an auxiliary loss
            aux_loss_multiplier = kwargs.get("aux_loss_multiplier", 20)
            if ys is not None:
                c = self.encoder_y.forward(xs)
                with pyro.poutine.scale(scale=aux_loss_multiplier):
                    pyro.sample(
                        "y_aux", dist.Normal(c, self.reg_sig).to_event(1), obs=ys)

    def guide_aux(self, xs, ys=None, **kwargs):
        """
        Dummy guide function to accompany model_aux
        """
        pass

    def set_regressor(self, reg_net: Type[torch.nn.Module]) -> None:
        """
        Sets a user-defined regression network
        """
        self.encoder_y = reg_net

    def regressor(self,
                  x_new: torch.Tensor,
                  **kwargs: int) -> torch.Tensor:
        """
        Applies trained regressor to new data

        Args:
            x_new:
                Input data for the regressor part of trained ss-reg-VAE.
                The new data must have the same dimensions
                (images height x width or spectra length) as the one used
                for training.
            kwargs:
                Batch size as 'batch_size' (for encoding large volumes of data)
        """
        def regress(x_i) -> torch.Tensor:
            with torch.no_grad():
                predicted = self.encoder_y(x_i)
            return predicted.cpu()

        x_new = init_dataloader(x_new, shuffle=False, **kwargs)
        y_predicted = []
        for (x_i,) in x_new:
            y_predicted.append(regress(x_i.to(self.device)))
        return torch.cat(y_predicted)

    def encode(self,
               x_new: torch.Tensor,
               y: Optional[torch.Tensor] = None,
               **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network. The output is
        a tuple with mean and standard deviations of the encoded distributions and
        a predicted continuous variable value.

        Args:
            x_new:
                Data to encode. The new data must have
                the same dimensions (images height and width or spectra length)
                as the one used for training.
            y:
                Vector with a continuous variable(s) for each sample in x_new.
                If not provided, the ss-reg-iVAE's regressor will be used to obtain it.
            kwargs:
                Batch size as 'batch_size' (for encoding large volumes of data)
        """
        if y is None:
            y = self.regressor(x_new, **kwargs)
        z = self._encode(x_new, y, **kwargs)
        z_loc, z_scale = z.split(self.z_dim, 1)
        return z_loc, z_scale, y

    def decode(self, z: torch.Tensor, y: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Decodes a batch of latent coordinates using a trained generator (decoder) network

        Args:
            z: Latent coordinates (without rotational and translational parts)
            y: Vector with continuous variable(s) for each sample in z
            kwargs: Batch size as 'batch_size'
        """
        z = torch.cat([z.to(self.device), y.to(self.device)], -1)
        loc = self._decode(z, **kwargs)
        return loc.view(-1, *self.data_dim)

    def manifold2d(self, d: int, y: torch.Tensor, plot: bool = True,
                   **kwargs: Union[str, int, float]) -> torch.Tensor:
        """
        Returns a learned latent manifold in the data space

        Args:
            d: Grid size
            y: Conditional vector
            plot: Plots the generated manifold (Default: True)
            kwargs: Keyword arguments include custom min/max values
                    for grid boundaries passed as 'z_coord'
                    (e.g. z_coord = [-3, 3, -3, 3]), 'angle' and
                    'shift' to condition a generative model on, and plot parameters
                    ('padding', 'pad_value', 'cmap', 'origin', 'ylim')
        """
        z, (grid_x, grid_y) = generate_latent_grid(d, **kwargs)
        y = y.unsqueeze(1) if 0 < y.ndim < 2 else y
        y = y.expand(z.shape[0], *y.shape[1:])
        loc = self.decode(z, y, **kwargs)
        if plot:
            if self.ndim == 2:
                plot_img_grid(
                    loc, d,
                    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                    **kwargs)
            elif self.ndim == 1:
                plot_spect_grid(loc, d, **kwargs)
        return loc
