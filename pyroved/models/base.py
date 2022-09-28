"""
base.py
=========

Variational encoder-decoder base class

Created by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""

from typing import Tuple, Type, Union, List
from abc import abstractmethod

import torch
import torch.nn as nn

from ..utils import init_dataloader, transform_coordinates, generate_grid, _to_device

tt = torch.tensor


class baseVAE(nn.Module):
    """Base class for regular and invriant variational encoder-decoder models.

    Args:
        data_dim:
            Dimensionality of the input data; use (height x width) for images
            or (length,) for spectra.
        invariances:
            List with invariances to enforce. For 2D systems, `r` enforces
            rotational invariance, `t` enforces invariance to
            translations, `sc` enforces a scale invariance, and
            invariances=None corresponds to vanilla VAE.
            For 1D systems, 't' enforces translational invariance and
            invariances=None is vanilla VAE

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
    """
    def __init__(self, *args, **kwargs: str):
        super(baseVAE, self).__init__()
        data_dim, invariances = args
        # Set device
        self.device = kwargs.get(
            "device", 'cuda' if torch.cuda.is_available() else 'cpu')
        # Set dimensionality
        self.ndim = len(data_dim)
        # Set invariances to enforce (number and type)
        if invariances is None:
            coord = 0
        else:
            coord = len(invariances)
            if self.ndim == 1:
                if coord > 1 or invariances[0] != 't':
                    raise ValueError(
                        "For 1D data, the only invariance to enforce "
                        "is translation ('t')")
            if 't' in invariances and self.ndim == 2:
                coord = coord + 1
        self.coord = coord
        self.invariances = invariances
        # Set coordiante grid
        if self.coord > 0:
            self.grid = generate_grid(data_dim).to(self.device)
        # Prior "belief" about the degree of translational disorder
        if self.coord > 0 and 't' in self.invariances:
            dx_pri = tt(kwargs.get("dx_prior", 0.1))
            dy_pri = kwargs.get("dy_prior", dx_pri.clone())
            self.t_prior = (tt([dx_pri, dy_pri]) if self.ndim == 2
                            else dx_pri).to(self.device)
        # Prior "belief" about the degree of scale disorder
        if self.coord > 0 and 's' in self.invariances:
            self.sc_prior = tt(kwargs.get("sc_prior", 0.1)).to(self.device)
        # Encoder and decoder (None by default)
        self.encoder_z = None
        self.decoder = None

    @abstractmethod
    def model(self, *args, **kwargs):
        """Pyro's model"""

        raise NotImplementedError

    @abstractmethod
    def guide(self, *args, **kwargs):
        """Pyro's guide"""

        raise NotImplementedError

    def _split_latent(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Split latent vector into parts associated with
        coordinate transformations and image content
        """
        # For 1D, there is only a translation
        if self.ndim == 1:
            dx = z[:, 0:1]
            z = z[:, 1:]
            return None, dx, None, z
        phi = tt(0).to(self.device)
        dx = tt(0).to(self.device)
        sc = tt(1).to(self.device)
        if 'r' in self.invariances:
            phi = z[:, 0]
            z = z[:, 1:]
        if 't' in self.invariances:
            dx = z[:, :2]
            z = z[:, 2:]
        if 's' in self.invariances:
            sc = sc + self.sc_prior * z[:, 0]
            z = z[:, 1:]
        return phi, dx, sc, z

    def _encode(
        self,
        *input_args: Tuple[Union[torch.Tensor, List[torch.Tensor]]],
        device: str = None,
        **kwargs: int
    ) -> torch.Tensor:
        """Encodes data using a trained inference (encoder) network
        in a batch-by-batch fashion."""

        device = self.device if device is None else device

        def inference(x: Tuple[torch.Tensor]) -> torch.Tensor:
            x = _to_device(x)
            with torch.no_grad():
                encoded = self.encoder_z(x)
            encoded = torch.cat(encoded, -1).cpu()
            return encoded

        loader = init_dataloader(*input_args, shuffle=False, **kwargs)
        z_encoded = []
        for x in loader:
            z_encoded.append(inference(x))
        return torch.cat(z_encoded)

    def _decode(self, z_new: torch.Tensor, device: str = None,
                **kwargs: int) -> torch.Tensor:
        """Decodes latent coordinates in a batch-by-batch fashion."""
        
        device = self.device if device is None else device

        def generator(z: List[torch.Tensor]) -> torch.Tensor:
            with torch.no_grad():
                loc = self.decoder(*z)
            return loc.cpu()

        z_new = init_dataloader(z_new, shuffle=False, **kwargs)
        if self.invariances:
            grid = self.grid
            a = kwargs.get("angle", tt(0.)).to(device)
            t = kwargs.get("shift", tt(0.)).to(device)
            s = kwargs.get("scale", tt(1.)).to(device)
            grid = transform_coordinates(
                grid.unsqueeze(0), a.unsqueeze(0),
                t.unsqueeze(0), s.unsqueeze(0))
            grid = grid.squeeze()
        x_decoded = []
        for z in z_new:
            if self.invariances:
                z = [grid.expand(z[0].shape[0], *grid.shape)] + z
            x_decoded.append(generator(z))
        return torch.cat(x_decoded)

    def set_encoder(self, encoder_net: Type[torch.nn.Module]) -> None:
        """Sets a user-defined encoder neural network."""

        self.encoder_z = encoder_net.to(self.device)

    def set_decoder(self, decoder_net: Type[torch.nn.Module]) -> None:
        """Sets a user-defined decoder neural network."""

        self.decoder = decoder_net.to(self.device)

    def save_weights(self, filepath: str) -> None:
        """Saves trained weights of encoder(s) and decoder."""

        torch.save(self.state_dict(), filepath + '.pt')

    def load_weights(self, filepath: str) -> None:
        """Loads saved weights of encoder(s) and decoder."""

        weights = torch.load(filepath, map_location=self.device)
        self.load_state_dict(weights)
