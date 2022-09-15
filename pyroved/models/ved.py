"""
ved.py
=========

Variational encoder-decoder model (input and output are different)

Created by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""
from typing import Tuple, Union, List

import pyro
import pyro.distributions as dist
import torch

from .base import baseVAE
from ..nets import convEncoderNet, convDecoderNet
from ..utils import (generate_latent_grid, get_sampler,
                     init_dataloader, plot_img_grid, plot_spect_grid,
                     set_deterministic_mode)


class VED(baseVAE):
    """
    Variational encoder-decoder model where the inputs and outputs are not identical.
    This model can be used for realizing im2spec and spec2im type of models where
    1D spectra are predicted from image data and vice versa.

    Args:
        input_dim:
            Dimensionality of the input data; use (h x w) for images
            or (length,) for spectra.
        output_dim:
            Dimensionality of the input data; use (h x w) for images
            or (length,) for spectra. Doesn't have to match the input data.
        input_channels:
            Number of input channels (Default: 1)
        output_channels:
            Number of output channels (Default: 1)
        latent_dim:
            Number of latent dimensions.
        hidden_dim_e:
            Number of hidden units (convolutional filters) for each layer in
            of the decoder NN. Defaults to [(32,), (64, 64), (128, 128)]
        hidden_dim_d:
            Number of hidden units (convolutional filters) for each layer in
            of the decoder NN. Defaults to [(128, 128), (64, 64), (32,)]
        activation:
            activation:
            Non-linear activation for inner layers of encoder and decoder.
            The available activations are ReLU ('relu'), leaky ReLU ('lrelu'),
            hyberbolic tangent ('tanh'), softplus ('softplus'), and GELU ('gelu').
            (The default is 'lrelu').
        batchnorm:
            Batch normalization attached to each convolutional layer
            after non-linear activation (except for layers with 1x1 filters)
            in the encoder and decoder NNs (Default: False)
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
            Additional keyword argument is *decoder_sig* for setting sigma
            in the decoder's sampler when it is chosen to be a "gaussian".

    Examples:

    Initialize a VED model for predicting 1D spectra from 2D images

    >>> # Initialize model
    >>> input_dim = (32, 32) # image height and width
    >>> output_dim = (1000,) # spectrum length
    >>> ved = VED(input_dim, output_dim, latent_dim=2)
    >>> # Initialize trainer
    >>> trainer = pv.trainers.SVItrainer(ved)
    >>> # Train for 100 epochs
    >>> for _ in range(100):
    >>>     trainer.step(train_loader)
    >>>     trainer.print_statistics()
    >>> # Visualize the learned latent manifold
    >>> ved.manifold2d(d=6, ylim=[0., .8], cmap='viridis');
    >>> # Make a prediction (image -> spectrum) on new data
    >>> pred_mean, pred_sd = ved.predict(new_inputs)
    """
    def __init__(self,
                 input_dim: Tuple[int],
                 output_dim: Tuple[int],
                 input_channels: int = 1,
                 output_channels: int = 1,
                 latent_dim: int = 2,
                 hidden_dim_e: List[int] = None,
                 hidden_dim_d: List[int] = None,
                 activation: str = "lrelu",
                 batchnorm: bool = False,
                 sampler_d: str = "bernoulli",
                 sigmoid_d: bool = True,
                 seed: int = 1,
                 **kwargs: float
                 ) -> None:
        """
        Initializes VED's modules and parameters
        """
        super(VED, self).__init__(output_dim, None, **kwargs)
        pyro.clear_param_store()
        set_deterministic_mode(seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ndim = len(output_dim)
        self.encoder_z = convEncoderNet(
            input_dim, latent_dim, input_channels,
            hidden_dim_e, batchnorm, activation)
        self.decoder = convDecoderNet(
            latent_dim, output_dim, output_channels,
            hidden_dim_d, batchnorm, activation, sigmoid_d)
        self.sampler_d = get_sampler(sampler_d, **kwargs)
        self.z_dim = latent_dim
        self.to(self.device)

    def model(self,
              x: torch.Tensor = None,
              y: torch.Tensor = None,
              **kwargs: float) -> None:
        """
        Defines the model p(y|z)p(z)
        """
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        # KLD scale factor (see e.g. https://openreview.net/pdf?id=Sy2fzU9gl)
        beta = kwargs.get("scale_factor", 1.)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            with pyro.poutine.scale(scale=beta):
                z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc = self.decoder(z)
            # score against actual images
            pyro.sample(
                "obs", self.sampler_d(loc.flatten(1)).to_event(1),
                obs=y.flatten(1))

    def guide(self,
              x: torch.Tensor = None,
              y: torch.Tensor = None,
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
                pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

    def encode(self, x_new: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained encoder network. The output is
        a tuple with means and standard deviations of the encoded distributions.

        Args:
            x_new:
                Data to encode with a trained VED. The new data must have
                the same dimensions (images height and width or spectra length)
                as the one used for training.
            kwargs:
                Batch size as 'batch_size' (for encoding large volumes of data)
        """
        self.eval()
        z = self._encode(x_new, **kwargs)
        z_loc, z_scale = z.split(self.z_dim, 1)
        return z_loc, z_scale

    def decode(self,
               z: torch.Tensor,
               **kwargs: int) -> torch.Tensor:
        """
        Decodes a batch of latent coordnates into the target space using
        a trained decoder network.

        Args:
            z: Latent coordinates
        """
        self.eval()
        z = z.to(self.device)
        loc = self._decode(z, **kwargs)
        return loc

    def predict(self, x_new: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """Forward prediction (encode -> sample -> decode)"""

        def forward_(x_i) -> torch.Tensor:
            with torch.no_grad():
                encoded = self.encoder_z(x_i)
                encoded = torch.cat(encoded, -1)
                z_mu, z_sig = encoded.split(self.z_dim, 1)
                z_samples = dist.Normal(z_mu, z_sig).rsample(sample_shape=(30,))
                y = torch.cat([self.decoder(z)[None] for z in z_samples])
            return y.mean(0).cpu(), y.std(0).cpu()

        x_new = init_dataloader(x_new, shuffle=False, **kwargs)
        prediction_mu, prediction_sd = [], []
        for (x_i,) in x_new:
            y_mu, y_sd = forward_(x_i.to(self.device))
            prediction_mu.append(y_mu)
            prediction_sd.append(y_sd)
        return torch.cat(prediction_mu), torch.cat(prediction_sd)

    def manifold2d(self, d: int, plot: bool = True,
                   **kwargs: Union[str, int]) -> torch.Tensor:
        """
        Plots a learned latent manifold in the image space

        Args:
            d: Grid size
            plot: Plots the generated manifold (Default: True)
            kwargs: Keyword arguments include custom min/max values for grid
                    boundaries passed as 'z_coord' (e.g. z_coord = [-3, 3, -3, 3])
                    and plot parameters ('padding', 'pad_value', 'cmap', 'origin', 'ylim')
        """
        self.eval()
        z, (grid_x, grid_y) = generate_latent_grid(d, **kwargs)
        z = z.to(self.device)
        with torch.no_grad():
            loc = self.decoder(z).cpu()
        if plot:
            if self.ndim == 2:
                plot_img_grid(
                    loc, d,
                    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                    **kwargs)
            elif self.ndim == 1:
                plot_spect_grid(loc, d, **kwargs)
        return loc
