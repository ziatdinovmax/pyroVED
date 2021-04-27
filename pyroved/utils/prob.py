from typing import Type
import pyro.distributions as dist


def get_sampler(
    sampler: str, normal_dist_scale: float = 0.5
) -> Type[dist.Distribution]:
    """Gets a sampler for VAE's decoder.

    [description]

    Parameters
    ----------
    sampler : {'bernoulli', 'continuous_bernoulli', 'gaussian'}
        [description]
    normal_dist_scale : {float}, optional
        Only used in the case of a Gaussian distribution. This is the scale
        parameter for the Gaussian distribution. (The default is 0.5).

    Returns
    -------
    Type[pyro.distributions.Distribution]

    Raises
    ------
    KeyError
        If the provided sampler key is not a valid distribution.
    """

    samplers = {
        "bernoulli": lambda x: dist.Bernoulli(x, validate_args=False),
        "continuous_bernoulli": lambda x: dist.ContinuousBernoulli(x),
        "gaussian": lambda x: dist.Normal(x, normal_dist_scale)
    }

    if sampler not in samplers.keys():
        raise KeyError(
            "Select between the following decoder "
            "samplers: {}".format(list(samplers.keys()))
        )

    return samplers[sampler]
