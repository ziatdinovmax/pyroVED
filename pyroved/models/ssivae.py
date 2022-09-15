"""
ssivae.py
=========

Semi-supervised variational autoencoder for data
with rotational, positional and scale disorders

Created by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""
from typing import Optional, Tuple, Union, Type, List

import pyro
import pyro.distributions as dist
import torch

from .base import baseVAE
from ..nets import fcDecoderNet, fcEncoderNet, sDecoderNet, fcClassifierNet
from ..utils import (get_sampler, plot_img_grid,
                     plot_spect_grid, set_deterministic_mode, to_onehot,
                     transform_coordinates, init_dataloader, generate_latent_grid,
                     generate_latent_grid_traversal, plot_grid_traversal)

tt = torch.tensor


class ssiVAE(baseVAE):
    """
    Semi-supervised variational autoencoder with the enforcement
    of rotational, translational, and scale invariances. It allows performing
    a classification of image/spectral data given a small number of examples
    even in the presence of a distribution shift between the labeled and unlabeled parts.

    Args:
        data_dim:
            Dimensionality of the input data; use (h x w) for images
            or (length,) for spectra.
        latent_dim:
            Number of latent dimensions.
        num_classes:
            Number of classes in the classification scheme
        invariances:
            List with invariances to enforce. For 2D systems, `r` enforces
            rotational invariance, `t` enforces invariance to
            translations, `sc` enforces a scale invariance, and
            invariances=None corresponds to vanilla VAE.
            For 1D systems, 't' enforces translational invariance and
            invariances=None is vanilla VAE
        hidden_dim_e:
            List with the number of hidden units in each layer
            of encoder (inference network). Defaults to [128, 128].
        hidden_dim_d:
            List with the number of hidden units in each layer
            of decoder (generator network). Defaults to [128, 128].
        hidden_dim_cls:
            List with the number of hidden units of each layer of classifier.
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

    Examples:

    Initialize a VAE model with rotational invariance for
    semi-supervised learning of the dataset that has 10 classes

    >>> # Initialize ssVAE
    >>> data_dim = (28, 28)
    >>> ssvae = ssiVAE(data_dim, latent_dim=2, num_classes=10, invariances=['r'])
    >>> # Initialize auxillary-SVI trainer
    >>> trainer = pv.trainers.auxSVItrainer(ssvae, task='classification')
    >>> # Get dataloaders
    >>> loader_unlabeled, loader_labeled, loader_val = pv.utils.init_ssvae_dataloaders(
    >>>     X_unlabeled, (X_labeled, y_labels), (X_val, y_val))
    >>> # Train for 100 epochs:
    >>> for e in range(100):
    >>>     trainer.step(loader_unlabeled, loader_labeled, loader_val, aux_loss_multiplier=50)
    >>>     trainer.print_statistics()
    >>> # Plot traversals of the learned latent manifolds
    >>> for i in range(2):
    >>>     ssvae.manifold_traversal(8, i, cmap='viridis')
    """
    def __init__(self,
                 data_dim: Tuple[int],
                 latent_dim: int,
                 num_classes: int,
                 invariances: List[str] = None,
                 hidden_dim_e: List[int] = None,
                 hidden_dim_d: List[int] = None,
                 hidden_dim_cls: List[int] = None,
                 activation: str = "tanh",
                 sampler_d: str = "bernoulli",
                 sigmoid_d: bool = True,
                 seed: int = 1,
                 **kwargs: Union[str, float]
                 ) -> None:
        """
        Initializes ss-iVAE parameters
        """
        args = (data_dim, invariances)
        super(ssiVAE, self).__init__(*args, **kwargs)
        pyro.clear_param_store()
        set_deterministic_mode(seed)

        self.data_dim = data_dim

        # Initialize z-Encoder neural network
        self.encoder_z = fcEncoderNet(
            data_dim, latent_dim+self.coord, num_classes,
            hidden_dim_e, activation, flat=False)

        # Initialize y-Encoder neural network
        self.encoder_y = fcClassifierNet(
            data_dim, num_classes, hidden_dim_cls, activation)

        # Initializes Decoder neural network
        dnet = sDecoderNet if 0 < self.coord < 5 else fcDecoderNet
        self.decoder = dnet(
            data_dim, latent_dim, num_classes, hidden_dim_d,
            activation, sigmoid_out=sigmoid_d, unflat=False)
        self.sampler_d = get_sampler(sampler_d, **kwargs)

        # Sets continuous and discrete dimensions
        self.z_dim = latent_dim + self.coord
        self.num_classes = num_classes

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
                if 'r' in self.invariances:
                    expdim = phi.shape[0]
                elif 't' in self.invariances:
                    expdim = dx.shape[0]
                elif 's' in self.invariances:
                    expdim = sc.shape[0]
                grid = self.grid.expand(expdim, *self.grid.shape)
                x_coord_prime = transform_coordinates(grid, phi, dx, sc)
            # sample label from the constant prior or observe the value
            alpha_prior = (torch.ones(batch_dim, self.num_classes, **specs) /
                           self.num_classes)
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)
            # Score against the parametrized distribution
            # p(x|y,z) = bernoulli(decoder(y,z)) or p(x|y,z) = gaussian(decoder(y,z), decoder_sig)
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
                alpha = self.encoder_y.forward(xs)
                with pyro.poutine.scale(scale=aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_aux(self, xs, ys=None, **kwargs):
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

        Args:
            x_new:
                Data to classify with a trained ss-iVAE. The new data must have
                the same dimensions (images height x width or spectra length)
                as the one used for training.
            kwargs:
                Batch size as 'batch_size' (for encoding large volumes of data)
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

        Args:
            x_new:
                Data to encode with a trained iVAE. The new data must have
                the same dimensions (images height and width or spectra length)
                as the one used for training.
            y:
                Classes as one-hot vectors for each sample in x_new. If not provided,
                the ss-iVAE's classifier will be used to predict the classes.
            kwargs:
                Batch size as 'batch_size' (for encoding large volumes of data)
        """
        if y is None:
            y = self.classifier(x_new, **kwargs)
        if y.ndim < 2:
            y = to_onehot(y, self.num_classes)
        z = self._encode(x_new, y, **kwargs)
        z_loc, z_scale = z.split(self.z_dim, 1)
        _, y_pred = torch.max(y, 1)
        return z_loc, z_scale, y_pred

    def decode(self, z: torch.Tensor, y: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Decodes a batch of latent coordinates

        Args:
            z: Latent coordinates (without rotational and translational parts)
            y: Classes as one-hot vectors for each sample in z
            kwargs: Batch size as 'batch_size'
        """
        z = torch.cat([z.to(self.device), y.to(self.device)], -1)
        loc = self._decode(z, **kwargs)
        return loc.view(-1, *self.data_dim)

    def manifold2d(self, d: int, plot: bool = True,
                   **kwargs: Union[str, int, float]) -> torch.Tensor:
        """
        Returns a learned latent manifold in the image space

        Args:
            d: Grid size
            plot: Plots the generated manifold (Default: True)
            kwargs: Keyword arguments include 'label' for class label (if any),
                    custom min/max values for grid boundaries passed as 'z_coord'
                    (e.g. z_coord = [-3, 3, -3, 3]), 'angle' and 'shift' to
                    condition a generative model one, and plot parameters
                    ('padding', 'padding_value', 'cmap', 'origin', 'ylim')
        """
        z, (grid_x, grid_y) = generate_latent_grid(d, **kwargs)
        cls = tt(kwargs.get("label", 0))
        if cls.ndim < 2:
            cls = to_onehot(cls.unsqueeze(0), self.num_classes)
        cls = cls.repeat(z.shape[0], 1)
        loc = self.decode(z, cls, **kwargs)
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
        Latent space traversal for continuous and discrete latent variables

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
        disc_dim = self.num_classes
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
