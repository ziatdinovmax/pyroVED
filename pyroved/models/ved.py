"""
ved.py
=========

Variational encoder-decoder model (input and output are different)

Created by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""
from typing import Tuple, Type, Union, List

import pyro
import pyro.distributions as dist
import torch

from ..nets import convEncoderNet, convDecoderNet
from ..utils import (generate_latent_grid, get_sampler,
                     init_dataloader, plot_img_grid, plot_spect_grid,
                     set_deterministic_mode)


class VED(torch.nn.Module):
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
            the first block of the encoder NN. The number of units in the
            consecutive blocks is defined as hidden_dim_e * n,
            where n = 2, 3, ..., n_blocks (Default: 32).
        hidden_dim_e:
            Number of hidden units (convolutional filters) for each layer in
            the first block of the decoder NN. The number of units in the
            consecutive blocks is defined as hidden_dim_e // n,
            where n = 2, 3, ..., n_blocks (Default: 32).
        num_layers_e:
            List with numbers of layers per each block of the encoder NN.
            Defaults to [1, 2, 2] if none is specified.
        num_layers_d:
            List with numbers of layers per each block of the decoder NN.
            Defaults to [2, 2, 1] if none is specified.
        activation:
            Non-linear activation for inner layers of encoder and decoder.
            The available activations are ReLU ('relu'), leaky ReLU ('lrelu'),
            hyberbolic tangent ('tanh'), and softplus ('softplus')
            The default activation is 'tanh'.
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

    Example:

    Initialize a VED model for predicting 1D spectra from 2D images

    >>> input_dim = (32, 32) # image height and width
    >>> output_dim = (16,) # spectrum length
    >>> ved = VED(input_dim, output_dim, latent_dim=2)
    """
    def __init__(self,
                 input_dim: Tuple[int],
                 output_dim: Tuple[int],
                 input_channels: int = 1,
                 output_channels: int = 1,
                 latent_dim: int = 2,
                 hidden_dim_e: int = 32,
                 hidden_dim_d: int = 96,
                 num_layers_e: List[int] = None,
                 num_layers_d: List[int] = None,
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
        super(VED, self).__init__()
        pyro.clear_param_store()
        set_deterministic_mode(seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ndim = len(output_dim)
        self.encoder_net = convEncoderNet(
            input_dim, input_channels, latent_dim,
            num_layers_e, hidden_dim_e,
            batchnorm, activation)
        self.decoder_net = convDecoderNet(
            latent_dim, output_dim, output_channels,
            num_layers_d, hidden_dim_d,
            batchnorm, activation, sigmoid_d)
        self.sampler_d = get_sampler(sampler_d, **kwargs)
        self.z_dim = latent_dim
        self.to(self.device)

    def model(self,
              x: torch.Tensor = None,
              y: torch.Tensor = None,
              **kwargs: float) -> torch.Tensor:
        """
        Defines the model p(y|z)p(z)
        """
        # register PyTorch module `decoder_net` with Pyro
        pyro.module("decoder_net", self.decoder_net)
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
            loc = self.decoder_net(z)
            # score against actual images
            pyro.sample(
                "obs", self.sampler_d(loc.flatten(1)).to_event(1),
                obs=y.flatten(1))

    def guide(self,
              x: torch.Tensor = None,
              y: torch.Tensor = None,
              **kwargs: float) -> torch.Tensor:
        """
        Defines the guide q(z|x)
        """
        # register PyTorch module `encoder_net` with Pyro
        pyro.module("encoder_net", self.encoder_net)
        # KLD scale factor (see e.g. https://openreview.net/pdf?id=Sy2fzU9gl)
        beta = kwargs.get("scale_factor", 1.)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder_net(x)
            # sample the latent code z
            with pyro.poutine.scale(scale=beta):
                pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

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

    def _encode(self,
                x_new: Union[torch.Tensor, torch.utils.data.DataLoader],
                **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network
        in a batch-by-batch fashion
        """
        def inference(x_i) -> torch.Tensor:
            with torch.no_grad():
                encoded = self.encoder_net(x_i)
            encoded = torch.cat(encoded, -1).cpu()
            return encoded

        if not isinstance(x_new, (torch.Tensor, torch.utils.data.DataLoader)):
            raise TypeError("Pass data as torch.Tensor or DataLoader object")
        if isinstance(x_new, torch.Tensor):
            x_new = init_dataloader(x_new, shuffle=False, **kwargs)
        z_encoded = []
        for (x_i,) in x_new:
            z_encoded.append(inference(x_i.to(self.device)))
        return torch.cat(z_encoded)

    def encode(self, x_new: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network
        (this is basically a wrapper for self._encode)
        """
        self.eval()
        z = self._encode(x_new)
        z_loc, z_scale = z.split(self.z_dim, 1)
        return z_loc, z_scale

    def _decode(self, z_new: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Decodes latent coordiantes in a batch-by-batch fashion
        """
        def generator(z: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                loc = self.decoder_net(*z)
            return loc.cpu()

        z_new = init_dataloader(z_new, shuffle=False, **kwargs)
        x_decoded = []
        for z in z_new:
            x_decoded.append(generator(z))
        return torch.cat(x_decoded)

    def decode(self,
               z: torch.Tensor,
               y: torch.Tensor = None,
               **kwargs: int) -> torch.Tensor:
        """
        Decodes a batch of latent coordnates
        """
        self.eval()
        z = z.to(self.device)
        loc = self._decode(z, **kwargs)
        return loc

    def predict(self, x_new: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """Forward prediction (encode -> sample -> decode)"""

        def forward_(x_i) -> torch.Tensor:
            with torch.no_grad():
                encoded = self.encoder_net(x_i)
                encoded = torch.cat(encoded, -1)
                z_mu, z_sig = encoded.split(self.z_dim, 1)
                z_samples = dist.Normal(z_mu, z_sig).rsample(sample_shape=(30,))
                y = torch.cat([self.decoder_net(z)[None] for z in z_samples])
            return y.mean(0).cpu(), y.std(0).cpu()

        if not isinstance(x_new, (torch.Tensor, torch.utils.data.DataLoader)):
            raise TypeError("Pass data as torch.Tensor or DataLoader object")
        if isinstance(x_new, torch.Tensor):
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
        """
        self.eval()
        z, (grid_x, grid_y) = generate_latent_grid(d)
        z = z.to(self.device)
        with torch.no_grad():
            loc = self.decoder_net(z).cpu()
        if plot:
            if self.ndim == 2:
                plot_img_grid(
                    loc, d,
                    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                    **kwargs)
            elif self.ndim == 1:
                plot_spect_grid(loc, d, **kwargs)
        return loc

    def save_weights(self, filepath: str) -> None:
        """
        Saves trained weights of encoder and decoder neural networks
        """
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath: str) -> None:
        """
        Loads saved weights of encoder and decoder neural networks
        """
        weights = torch.load(filepath, map_location=self.device)
        self.load_state_dict(weights)
