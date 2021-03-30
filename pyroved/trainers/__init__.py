"""
Wrappers for Pyro training loops with stochastic variational inference
"""
from .svi import SVItrainer
from .auxsvi import auxSVItrainer

__all__ = ['SVItrainer', 'auxSVItrainer']