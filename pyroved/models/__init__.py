"""
Variational autoencoder and encoder-decoder models
"""
from .ivae import iVAE
from .ssivae import ssiVAE
from .jivae import jiVAE
from .ved import VED

__all__ = ['iVAE', 'jiVAE', 'ssiVAE', 'VED']
