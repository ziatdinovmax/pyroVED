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

from ..utils import init_dataloader


class baseVAE(nn.Module):
    """
    Base class for variational autoencoder models
    """
    def __init__(self):
        super(baseVAE, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder_z = None
        self.decoder = None
        self.coord = 0
        self.grid = None

    @abstractmethod
    def model(self):
        """Pyro's model"""
        raise NotImplementedError

    @abstractmethod
    def guide(self):
        """Pyro's guide"""
        raise NotImplementedError

    def _encode(self,
                *input_args: Union[torch.Tensor, List[torch.Tensor]],
                **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network
        in a batch-by-batch fashion
        """
        def inference(x: Tuple[torch.Tensor]) -> torch.Tensor:
            x = torch.cat(x, -1).to(self.device)
            with torch.no_grad():
                encoded = self.encoder_z(x)
            encoded = torch.cat(encoded, -1).cpu()
            return encoded

        loader = init_dataloader(*input_args, shuffle=False, **kwargs)
        z_encoded = []
        for x in loader:
            z_encoded.append(inference(x))
        return torch.cat(z_encoded)

    def _decode(self, z_new: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Decodes latent coordiantes in a batch-by-batch fashion
        """
        def generator(z: List[torch.Tensor]) -> torch.Tensor:
            with torch.no_grad():
                loc = self.decoder(*z)
            return loc.cpu()

        z_new = init_dataloader(z_new, shuffle=False, **kwargs)
        x_decoded = []
        for z in z_new:
            if self.coord > 0:
                z = [self.grid.expand(z[0].shape[0], *self.grid.shape)] + z
            x_decoded.append(generator(z))
        return torch.cat(x_decoded)

    def set_encoder(self, encoder_net: Type[torch.nn.Module]) -> None:
        """
        Sets a user-defined encoder neural network
        """
        self.encoder_z = encoder_net.to(self.device)

    def set_decoder(self, decoder_net: Type[torch.nn.Module]) -> None:
        """
        Sets a user-defined decoder neural network
        """
        self.decoder = decoder_net.to(self.device)

    def save_weights(self, filepath: str) -> None:
        """
        Saves trained weights of encoder(s) and decoder
        """
        torch.save(self.state_dict(), filepath + '.pt')

    def load_weights(self, filepath: str) -> None:
        """
        Loads saved weights of encoder(s) and decoder
        """
        weights = torch.load(filepath, map_location=self.device)
        self.load_state_dict(weights)