"""
sstrvae.py
=========

Semi-supervised variational autoencoder for data
with positional (rotation+translation) disorder

Created by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""
from typing import Optional, Tuple, Union, Type

import pyro
import pyro.distributions as dist
import torch
import torch.tensor as tt

from .base import baseVAE
from ..nets import fcDecoderNet, fcEncoderNet, sDecoderNet, fcClassifierNet
from ..utils import (generate_grid, get_sampler, plot_img_grid,
                     plot_spect_grid, set_deterministic_mode, to_onehot,
                     transform_coordinates, init_dataloader, generate_latent_grid)


class sstrVAE(baseVAE):
    """
    Semi-supervised variational autoencoder with rotational and/or translational invariance

    Args:
        data_dim:
            Dimensionality of the input data; use (h x w) for images
            or (length,) for spectra.
        latent_dim:
            Number of latent dimensions.
        num_classes:
            Number of classes in the classification scheme
        coord:
            For 2D systems, *coord=0* is vanilla VAE, *coord=1* enforces
            rotational invariance, *coord=2* enforces invariance to translations,
            and *coord=3* enforces both rotational and translational invariances.
            For 1D systems, *coord=0* is vanilla VAE and *coord>0* enforces
            transaltional invariance.
        hidden_dim_e:
            Number of hidden units per each layer in encoder (inference network).
        hidden_dim_d:
            Number of hidden units per each layer in decoder (generator network).
        hidden_dim_cls:
            Number of hidden units ("neurons") in each layer of classifier
        num_layers_e:
            Number of layers in encoder (inference network).
        num_layers_d:
            Number of layers in decoder (generator network).
        num_layers_cls:
            Number of layers in classifier
        sampler_d:
            Decoder sampler, as defined as p(x|z) = sampler(decoder(z)).
            The available samplers are 'bernoulli', 'continuous_bernoulli',
            and 'gaussian' (Default: 'bernoulli').
        sigmoid_d:
            Sigmoid activation for the decoder output (Default: True)
        seed:
            Seed used in torch.manual_seed(seed) and
            torch.cuda.manual_seed_all(seed)
        kwargs:
            Additional keyword arguments are *dx_prior* and *dy_prior* for setting
            a translational prior(s), and *decoder_sig* for setting sigma
            in the decoder's sampler when it is set to "gaussian".

    Example:

    Initialize a VAE model with rotational invariance for
    semisupervised learning of the dataset that has 10 classes

    >>> data_dim = (28, 28)
    >>> ssvae = sstrVAE(data_dim, latent_dim=2, num_classes=10, coord=1)
    """
    def __init__(self,
                 data_dim: Tuple[int],
                 latent_dim: int,
                 num_classes: int,
                 coord: int = 3,
                 hidden_dim_e: int = 128,
                 hidden_dim_d: int = 128,
                 hidden_dim_cls: int = 128,
                 num_layers_e: int = 2,
                 num_layers_d: int = 2,
                 num_layers_cls: int = 2,
                 sampler_d: str = "bernoulli",
                 sigmoid_d: bool = True,
                 seed: int = 1,
                 **kwargs: float
                 ) -> None:
        """
        Initializes sstrVAE parameters
        """
        super(sstrVAE, self).__init__()
        pyro.clear_param_store()
        set_deterministic_mode(seed)
        if coord not in [0, 1, 2, 3]:
            raise ValueError("'coord' argument must be 0, 1, 2 or 3")
        self.ndim = len(data_dim)
        if self.ndim == 1 and coord > 0:
            coord = 1
        self.data_dim = data_dim
        self.encoder_z = fcEncoderNet(
            data_dim, latent_dim+coord, num_classes,
            hidden_dim_e, num_layers_e, flat=False)
        self.encoder_y = fcClassifierNet(
            data_dim, num_classes, hidden_dim_cls, num_layers_cls)
        dnet = sDecoderNet if coord in [1, 2, 3] else fcDecoderNet
        self.decoder = dnet(
            data_dim, latent_dim, num_classes, hidden_dim_d,
            num_layers_d, sigmoid_out=sigmoid_d, unflat=False)
        self.sampler_d = get_sampler(sampler_d, **kwargs)
        self.z_dim = latent_dim + coord
        self.num_classes = num_classes
        self.coord = coord
        self.grid = generate_grid(data_dim).to(self.device)
        dx_pri = tt(kwargs.get("dx_prior", 0.1))
        dy_pri = kwargs.get("dy_prior", dx_pri.clone())
        t_prior = tt([dx_pri, dy_pri]) if self.ndim == 2 else dx_pri
        self.t_prior = t_prior.to(self.device)
        self.to(self.device)

    def model(self,
              xs: torch.Tensor,
              ys: Optional[torch.Tensor] = None,
              **kwargs: float) -> torch.Tensor:
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
                phi, dx, zs = self.split_latent(zs)
                if torch.sum(dx.abs()) != 0:
                    dx = (dx * self.t_prior).unsqueeze(1)
                # transform coordinate grid
                if self.ndim > 1:
                    expdim = dx.shape[0] if self.coord > 1 else phi.shape[0]
                else:
                    expdim = dx.shape[0]
                grid = self.grid.expand(expdim, *self.grid.shape)
                x_coord_prime = transform_coordinates(grid, phi, dx)
            # sample label from the constant prior or observe the value
            alpha_prior = (torch.ones(batch_dim, self.num_classes, **specs) /
                           self.num_classes)
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)
            # Score against the parametrized distribution
            # p(x|y,z) = bernoulli(decoder(y,z))
            d_args = (x_coord_prime, [zs, ys]) if self.coord else ([zs, ys],)
            loc = self.decoder(*d_args)
            loc = loc.view(*ys.shape[:-1], -1)
            pyro.sample("x", self.sampler_d(loc).to_event(1), obs=xs)

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
                alpha = self.encoder_y(xs)
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))
            # sample (and score) the latent vector with the variational
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
        if self.ndim == 1:
            dx = zs[:, 0:1]
            zs = zs[:, 1:]
            return None, dx, zs.view(*zdims)
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
        zs = zs.view(*zdims)
        return phi, dx, zs

    def model_classify(self, xs: torch.Tensor,
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
                alpha = self.encoder_y.forward(xs)
                with pyro.poutine.scale(scale=aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None, **kwargs):
        """
        Dummy guide function to accompany model_classify
        """
        pass

    def set_classifier(self, cls_net: Type[torch.nn.Module]) -> None:
        """
        Sets a user-defined classification network
        """
        self.encoder_y = cls_net

    def classifier(self,
                   x_new: torch.Tensor,
                   **kwargs: int) -> torch.Tensor:
        """
        Classifies data
        """
        def classify(x_i) -> torch.Tensor:
            with torch.no_grad():
                alpha = self.encoder_y(x_i)
            _, predicted = torch.max(alpha.data, 1)
            return predicted.cpu()

        x_new = init_dataloader(x_new, shuffle=False, **kwargs)
        y_predicted = []
        for (x_i,) in x_new:
            y_predicted.append(classify(x_i.to(self.device)))
        return torch.cat(y_predicted)

    def encode(self,
               x_new: torch.Tensor,
               y: Optional[torch.Tensor] = None,
               **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network
        """
        if y is None:
            y = self.classifier(x_new)
        if y.ndim < 2:
            y = to_onehot(y, self.num_classes)
        z = self._encode(x_new, y, **kwargs)
        z_loc, z_scale = z.split(self.z_dim, 1)
        _, y_pred = torch.max(y, 1)
        return z_loc, z_scale, y_pred

    def decode(self, z: torch.Tensor, y: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Decodes a batch of latent coordinates
        """
        z = torch.cat([z.to(self.device), y.to(self.device)], -1)
        loc = self._decode(z, **kwargs)
        return loc.view(-1, *self.data_dim)

    def manifold2d(self, d: int, plot: bool = True,
                   **kwargs: Union[str, int]) -> torch.Tensor:
        """
        Returns a learned latent manifold in the image space
        """
        cls = tt(kwargs.get("label", 0))
        if cls.ndim < 2:
            cls = to_onehot(cls.unsqueeze(0), self.num_classes)
        z, (grid_x, grid_y) = generate_latent_grid(d)
        z = z.to(self.device)
        z = torch.cat([z, cls.repeat(z.shape[0], 1)], dim=-1)
        z = [z]
        if self.coord:
            grid = [self.grid.expand(z[0].shape[0], *self.grid.shape)]
            z = grid + z
        with torch.no_grad():
            loc = self.decoder(*z).cpu()
        loc = loc.view(-1, *self.data_dim)
        if plot:
            if self.ndim == 2:
                plot_img_grid(
                    loc, d,
                    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                    **kwargs)
            elif self.ndim == 1:
                plot_spect_grid(loc, d, **kwargs)
        return loc