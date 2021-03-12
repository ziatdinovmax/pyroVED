from typing import Optional, Tuple, Union

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.tensor as tt
from scipy.stats import norm

from ..nets import fcDecoderNet, fcEncoderNet, sDecoderNet, fcClassifierNet
from ..utils import (generate_grid, get_sampler, plot_img_grid,
                     plot_spect_grid, set_deterministic_mode, to_onehot,
                     transform_coordinates)


class sstrVAE(nn.Module):
    """
    Semi-supervised variational autoencoder (VAE) with rotational
    and/or translational invariance
    """
    def __init__(self,
                 in_dim: Tuple[int],
                 latent_dim: int,
                 num_classes: int,
                 coord: int = 3,
                 aux_loss_multiplier: int = 20,
                 hidden_dim_e: int = 128,
                 hidden_dim_d: int = 128,
                 hidden_dim_cls: int = 128,
                 num_layers_e: int = 2,
                 num_layers_d: int = 2,
                 num_layers_cls: int = 2,
                 sampler_d: str = "bernoulli",
                 seed: int = 1,
                 **kwargs
                 ) -> None:
        super(sstrVAE, self).__init__()
        pyro.clear_param_store()
        set_deterministic_mode(seed)
        if coord not in [0, 1, 2, 3]:
            raise ValueError("'coord' argument must be 0, 1, 2 or 3")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder_z = fcEncoderNet(
            in_dim, latent_dim+coord, num_classes,
            hidden_dim_e, num_layers_e, flat=False)
        self.encoder_y = fcClassifierNet(
            in_dim, num_classes, hidden_dim_cls, num_layers_cls)
        dnet = sDecoderNet if coord in [1, 2, 3] else fcDecoderNet
        self.decoder = dnet(
            in_dim, latent_dim, num_classes, hidden_dim_d,
            num_layers_d, unflat=False)
        self.sampler_d = get_sampler[sampler_d]
        self.z_dim = latent_dim + coord
        self.num_classes = num_classes
        self.coord = coord
        self.grid = generate_grid(in_dim).to(self.device)
        self.dx_prior = kwargs.get("dx_prior", 0.1)
        self.aux_loss_multiplier = aux_loss_multiplier
        self.to(self.device)

    def model(self,
              xs: torch.Tensor,
              ys: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Model of the generative process p(x|z,y)p(y)p(z)
        """
        pyro.module("ss_vae", self)
        batch_dim = xs.size(0)
        specs = dict(dtype=xs.dtype, device=xs.device)
        # pyro.plate enforces independence between variables in batches xs, ys
        with pyro.plate("data"):
            # sample the latent vector from the constant prior distribution
            prior_loc = torch.zeros(batch_dim, self.z_dim, **specs)
            prior_scale = torch.ones(batch_dim, self.z_dim, **specs)
            zs = pyro.sample(
                "z", dist.Normal(prior_loc, prior_scale).to_event(1))
            # split latent variable into parts for rotation and/or translation
            # and image content
            if self.coord > 0:
                phi, dx, zs = self.split_latent(zs)
                if torch.sum(dx) != 0:
                    dx = (dx * self.dx_prior).unsqueeze(1)
                # transform coordinate grid
                expdim = dx.shape[0] if self.coord > 1 else phi.shape[0]
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
              ys: Optional[torch.Tensor] = None) -> None:
        """
        Guide q(z|y,x)q(y|x)
        """
        pyro.module("ss_vae", self)
        with pyro.plate("data"):
            # sample and score the digit with the variational distribution
            # q(y|x) = categorical(alpha(x))
            if ys is None:
                alpha = self.encoder_y(xs)
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))
            # sample (and score) the latent vector with the variational
            # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            loc, scale = self.encoder_z([xs, ys])
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def split_latent(self, zs: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Split latent variable into parts with rotation and/or translation
        and image content
        """
        zdims = list(zs.shape)
        zdims[-1] = zdims[-1] - self.coord
        zs = zs.view(-1, zs.size(-1))
        phi, dx = tt(0), tt(0)
        # rotation + translation
        if self.coord == 3:
            phi = zs[:, 0] # encoded angle
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
                       ys: Optional[torch.Tensor] = None) -> None:
        """
        Models an auxiliary (supervised) loss
        """
        pyro.module("ss_vae", self)
        with pyro.plate("data"):
            # the extra term to yield an auxiliary loss
            if ys is not None:
                alpha = self.encoder_y.forward(xs)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None):
        """
        dummy guide function to accompany model_classify
        """
        pass

    def classifier(self,
                   x_new: torch.Tensor,
                   **kwargs: int) -> torch.Tensor:
        """
        Classifies images
        """
        def classify() -> torch.Tensor:
            with torch.no_grad():
                alpha = self.encoder_y(x_i.to(self.device))
            _, predicted = torch.max(alpha.data, 1)
            return predicted.cpu()

        num_batches = kwargs.get("num_batches", 1)
        batch_size = len(x_new) // num_batches
        y_predicted = []
        for i in range(num_batches):
            x_i = x_new[i*batch_size:(i+1)*batch_size]
            y_pred_i = classify()
            y_predicted.append(y_pred_i)
        x_i = x_new[(i+1)*batch_size:]
        if len(x_i) > 0:
            y_pred_i = classify()
            y_predicted.append(y_pred_i)
        return torch.cat(y_predicted)

    def _encode(self,
                x_new: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network
        in a batch-by-batch fashion
        """
        def inference() -> torch.Tensor:
            with torch.no_grad():
                encoded = self.encoder_z(torch.cat([x_i, y_i], dim=-1))
            encoded = torch.cat(encoded, -1).cpu()
            return encoded

        if y is None:
            y = self.classifier(x_new)
        if y.ndim == 1:
            y = to_onehot(y, self.num_classes)
        num_batches = kwargs.get("num_batches", 1)
        batch_size = len(x_new) // num_batches
        z_encoded = []
        for i in range(num_batches):
            x_i = x_new[i*batch_size:(i+1)*batch_size].to(self.device)
            y_i = y[i*batch_size:(i+1)*batch_size].to(self.device)
            z_encoded_i = inference()
            z_encoded.append(z_encoded_i)
        x_i = x_new[(i+1)*batch_size:].to(self.device)
        y_i = y[(i+1)*batch_size:].to(self.device)
        if len(x_i) > 0:
            z_encoded_i = inference()
            z_encoded.append(z_encoded_i)
        _, pred_labels = torch.max(y, 1)
        return torch.cat(z_encoded), pred_labels.cpu()

    def encode(self,
               x_new: torch.Tensor,
               y: Optional[torch.Tensor] = None,
               **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network
        """
        if isinstance(x_new, torch.utils.data.DataLoader):
            x_new = x_new.dataset.tensors[0]
        if isinstance(y, torch.utils.data.DataLoader):
            y = y.dataset.tensors[0]
        z, y_pred = self._encode(x_new, y, **kwargs)
        z_loc = z[:, :self.z_dim]
        z_scale = z[:, self.z_dim:]
        return z_loc, z_scale, y_pred

    def manifold2d(self, d: int, **kwargs: Union[str, int]) -> torch.Tensor:
        """
        Returns a learned latent manifold in the image space
        """
        cls = tt(kwargs.get("label", 0))
        cls = to_onehot(cls.unsqueeze(0), self.num_classes)
        grid_x = norm.ppf(torch.linspace(0.95, 0.05, d))
        grid_y = norm.ppf(torch.linspace(0.05, 0.95, d))
        loc_all = []
        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):
                z_sample = tt([xi, yi]).float().to(self.device).unsqueeze(0)
                z_sample = torch.cat([z_sample, cls], dim=-1)
                d_args = (self.grid.unsqueeze(0), z_sample) if self.coord > 0 else (z_sample,)
                loc = self.decoder(*d_args)
                loc_all.append(loc.detach().cpu())
        loc_all = torch.cat(loc_all)

        if self.ndim == 2:
            plot_img_grid(
                loc_all, d,
                extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                **kwargs)
        elif self.ndim == 1:
            plot_spect_grid(loc_all, d)

        return loc_all