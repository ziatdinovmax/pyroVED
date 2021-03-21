from typing import Optional, Tuple, Union, Type

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.tensor as tt

from ..nets import fcDecoderNet, jfcEncoderNet, sDecoderNet
from ..utils import (generate_grid, get_sampler, plot_img_grid,
                     plot_spect_grid, set_deterministic_mode,
                     transform_coordinates, to_onehot)


class jtrVAE(nn.Module):
    """
    Args:
        data_dim:
            Dimensionality of the input data; use (h x w) for images
            or (length,) for spectra.
        latent_dim:
            Number of continuous latent dimensions.
        discrete_dim:
            Number of discrete latent dimensions
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
        num_layers_e:
            Number of layers in encoder (inference network).
        num_layers_d:
            Number of layers in decoder (generator network).
        activation:
            Non-linear activation for inner layers of encoder and decoder.
            The available activations are ReLU ('relu'), leaky ReLU ('lrelu'),
            hyberbolic tangent ('tanh'), and softplus ('softplus')
            The default activation is 'tanh'.
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

    Initialize a joint VAE model with rotational invariance for 10 discrete classes

    >>> data_dim = (28, 28)
    >>> ssvae = jtrVAE(data_dim, latent_dim=2, discrete_dim=10, coord=1)
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
        Initializes trVAE's modules and parameters
        """
        super(jtrVAE, self).__init__()
        pyro.clear_param_store()
        set_deterministic_mode(seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ndim = len(data_dim)
        self.data_dim = data_dim
        if self.ndim == 1 and coord > 0:
            coord = 1
        self.encoder_net = jfcEncoderNet(
            data_dim, latent_dim+coord, discrete_dim, hidden_dim_e,
            num_layers_e, activation, softplus_out=True)
        if coord not in [0, 1, 2, 3]:
            raise ValueError("'coord' argument must be 0, 1, 2 or 3")
        dnet = sDecoderNet if coord in [1, 2, 3] else fcDecoderNet
        self.decoder_net = dnet(
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
              y: Optional[torch.Tensor] = None,
              **kwargs: float) -> torch.Tensor:
        """
        Defines the model p(x|z,c)p(z)p(c)
        """
        # register PyTorch module `decoder_net` with Pyro
        pyro.module("decoder_net", self.decoder_net)
        # KLD scale factor (see e.g. https://openreview.net/pdf?id=Sy2fzU9gl)
        beta = kwargs.get("scale_factor", 1.)
        reshape_ = torch.prod(tt(x.shape[1:])).item()
        bdim = x.shape[0]
        with pyro.plate("data"):
            # sample the continuous latent vector from the constant prior distribution
            z_loc = x.new_zeros(torch.Size((bdim, self.z_dim)))
            z_scale = x.new_ones(torch.Size((bdim, self.z_dim)))
            # sample discrete latent vector from the constant prior
            alpha = x.new_ones(torch.Size((bdim, self.discrete_dim))) / self.discrete_dim
            # sample from prior (value will be sampled by guide when computing ELBO)
            with pyro.poutine.scale(scale=beta):
                z = pyro.sample("latent_cont", dist.Normal(z_loc, z_scale).to_event(1))
                z_disc = pyro.sample("latent_disc", dist.OneHotCategorical(alpha))
            # split latent variable into parts for rotation and/or translation
            # and image content
            if self.coord > 0:
                phi, dx, z = self.split_latent(z.repeat(self.discrete_dim, 1))
                if torch.sum(dx) != 0:
                    dx = (dx * self.t_prior).unsqueeze(1)
                # transform coordinate grid
                grid = self.grid.expand(bdim*self.discrete_dim, *self.grid.shape)
                x_coord_prime = transform_coordinates(grid, phi, dx)
            # Continuous and discrete latent variables for the decoder
            z = [z, z_disc.reshape(-1, self.discrete_dim) if self.coord > 0 else z_disc]
            # decode the latent code z together with the transformed coordinates (if any)
            dec_args = (x_coord_prime, z) if self.coord else (z,)
            loc = self.decoder_net(*dec_args)
            # score against actual images/spectra
            loc = loc.view(*z_disc.shape[:-1], reshape_)
            pyro.sample(
                "obs", self.sampler_d(loc).to_event(1),
                obs=x.view(-1, reshape_))

    def guide(self,
              x: torch.Tensor,
              y: Optional[torch.Tensor] = None,
              **kwargs: float) -> torch.Tensor:
        """
        Defines the guide q(z,c|x)
        """
        # register PyTorch module `encoder_net` with Pyro
        pyro.module("encoder_net", self.encoder_net)
        # KLD scale factor (see e.g. https://openreview.net/pdf?id=Sy2fzU9gl)
        beta = kwargs.get("scale_factor", 1.)
        with pyro.plate("data"):
            # use the encoder to get the parameters used to define q(z,c|x)
            z_loc, z_scale, alpha = self.encoder_net(x)
            # sample the latent code z
            with pyro.poutine.scale(scale=beta):
                pyro.sample("latent_cont", dist.Normal(z_loc, z_scale).to_event(1))
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

    def set_encoder(self, encoder_net: Type[torch.nn.Module]) -> None:
        """
        Sets a user-defined encoder network
        """
        self.encoder_z = encoder_net

    def set_decoder(self, decoder_net: Type[torch.nn.Module]) -> None:
        """
        Sets a user-defined decoder network
        """
        self.decoder = decoder_net

    def _encode(self, x_new: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network
        in a batch-by-batch fashion
        """
        def inference() -> torch.Tensor:
            with torch.no_grad():
                encoded = self.encoder_net(x_i)
            encoded = torch.cat(encoded, -1).cpu()
            return encoded

        x_new = x_new.to(self.device)
        num_batches = kwargs.get("num_batches", 10)
        batch_size = len(x_new) // num_batches
        z_encoded = []
        for i in range(num_batches):
            x_i = x_new[i*batch_size:(i+1)*batch_size]
            z_encoded_i = inference()
            z_encoded.append(z_encoded_i)
        x_i = x_new[(i+1)*batch_size:]
        if len(x_i) > 0:
            z_encoded_i = inference()
            z_encoded.append(z_encoded_i)
        return torch.cat(z_encoded)

    def encode(self, x_new: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network
        """
        if isinstance(x_new, torch.utils.data.DataLoader):
            x_new = x_new.dataset.tensors[0]
        z = self._encode(x_new)
        z_loc = z[:, :self.z_dim]
        z_scale = z[:, self.z_dim:2*self.z_dim]
        alphas = z[:, 2*self.z_dim:]
        return z_loc, z_scale, alphas

    def decode(self, z: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Decodes a batch of latent coordnates
        """
        z = z.to(self.device)
        if y is not None:
            z = torch.cat([z, y.to(self.device)], -1)
        z = (z,)
        if self.coord > 0:
            z = (self.grid.expand(z[0].shape[0], *self.grid.shape),) + z
        with torch.no_grad():
            loc = self.decoder_net(*z)
        return loc

    def manifold2d(self, d: int, disc_idx: int = 0, plot: bool = True,
                   **kwargs: Union[str, int]) -> torch.Tensor:
        """
        Plots a learned latent manifold in the image space
        """
        cls = to_onehot(tt(disc_idx).unsqueeze(0), self.discrete_dim)
        grid_x = dist.Normal(0, 1).icdf(torch.linspace(0.95, 0.05, d))
        grid_y = dist.Normal(0, 1).icdf(torch.linspace(0.05, 0.95, d))
        loc_all = []
        for xi in grid_x:
            for yi in grid_y:
                z = tt([xi, yi]).float().to(self.device).unsqueeze(0)
                z = torch.cat([z, cls], dim=-1)
                d_args = (self.grid.unsqueeze(0), z) if self.coord > 0 else (z,)
                loc = self.decoder_net(*d_args)
                loc_all.append(loc.detach().cpu())
        loc_all = torch.cat(loc_all)
        loc_all = loc_all.view(-1, *self.data_dim)
        if plot:
            if self.ndim == 2:
                plot_img_grid(
                    loc_all, d,
                    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                    **kwargs)
            elif self.ndim == 1:
                plot_spect_grid(loc_all, d)
        return loc_all
