from typing import Type
import pyro.distributions as dist


def get_sampler(sampler: str, **kwargs) -> Type[dist.Distribution]:
    """
    Get a sampler for VAE's decoder
    """
    samplers = {
        "bernoulli": lambda x: dist.Bernoulli(x, validate_args=False),
        "continuous_bernoulli": lambda x: dist.ContinuousBernoulli(x),
        "gaussian": lambda x: dist.Normal(x, kwargs.get("decoder_sig", 0.5))
        }
    if sampler not in samplers.keys():
        raise NotImplementedError(
            "Select between the following decoder samplers: {}".format(list(samplers.keys())))
    return samplers[sampler]
