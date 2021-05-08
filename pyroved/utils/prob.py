from typing import Type
import pyro.distributions as dist


def get_sampler(
    sampler: str, **kwargs: float
     ) -> Type[dist.Distribution]:
    """Gets a sampler for VAE's decoder.

    Args:
        sampler: 'bernoulli', 'continuous_bernoulli', 'gaussian'

    Keyword Args:
        decoder_sig:
            The scale parameter for the Gaussian distribution (Defaults to 0.5)

    Returns:
        Pyro distribution object for the decoder sampling

    Raises:
        KeyError:
            If the provided sampler key is not a valid distribution.
    """

    samplers = {
        "bernoulli": lambda x: dist.Bernoulli(x, validate_args=False),
        "continuous_bernoulli": lambda x: dist.ContinuousBernoulli(x),
        "gaussian": lambda x: dist.Normal(x, kwargs.get("decoder_sig", 0.5))
        }

    if sampler not in samplers.keys():
        raise KeyError(
            "Select between the following decoder "
            "samplers: {}".format(list(samplers.keys()))
        )

    return samplers[sampler]
