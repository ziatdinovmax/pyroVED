import sys

import torch
import torch.tensor as tt
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
import pyro.infer as infer
from pyro.poutine.enum_messenger import EnumMessenger
import pytest
from numpy.testing import assert_equal, assert_

sys.path.append("../../")

from pyroved import models


def get_traces(model, x):
    guide_trace = pyro.poutine.trace(model.guide).get_trace(x)
    model_trace = pyro.poutine.trace(
        pyro.poutine.replay(model.model, trace=guide_trace)).get_trace(x)
    return guide_trace, model_trace


def get_enum_traces(model, x):
    guide_enum = EnumMessenger(first_available_dim=-2)
    model_enum = EnumMessenger()
    guide_ = guide_enum(
        infer.config_enumerate(model.guide, "parallel", expand=True))
    model_ = model_enum(model.model)
    guide_trace = poutine.trace(guide_, graph_type="flat").get_trace(x)
    model_trace = poutine.trace(
        pyro.poutine.replay(model_, trace=guide_trace),
        graph_type="flat").get_trace(x)
    return guide_trace, model_trace


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_trvae_sites_dims(data_dim, coord):
    x = torch.randn(*data_dim)
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.trVAE(data_dim[1:], coord=coord)
    guide_trace, model_trace = get_traces(model, x)
    assert_equal(model_trace.nodes["latent"]['value'].shape,
                 (data_dim[0], coord+2))
    assert_equal(guide_trace.nodes["latent"]['value'].shape,
                 (data_dim[0], coord+2))
    assert_equal(model_trace.nodes["obs"]['value'].shape,
                 (data_dim[0], torch.prod(tt(data_dim[1:])).item()))


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_trvae_sites_fn(data_dim, coord):
    x = torch.randn(*data_dim)
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.trVAE(data_dim[1:], coord=coord)
    guide_trace, model_trace = get_traces(model, x)
    assert_(isinstance(model_trace.nodes["latent"]['fn'].base_dist, dist.Normal))
    assert_(isinstance(guide_trace.nodes["latent"]['fn'].base_dist, dist.Normal))
    assert_(isinstance(model_trace.nodes["obs"]['fn'].base_dist, dist.Bernoulli))


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_jtrvae_cont_sites_dims(data_dim, coord):
    x = torch.randn(*data_dim)
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.jtrVAE(data_dim[1:], 2, 3, coord=coord)
    guide_trace, model_trace = get_enum_traces(model, x)
    assert_equal(model_trace.nodes["latent_cont"]['value'].shape,
                 (data_dim[0], coord+2))
    assert_equal(guide_trace.nodes["latent_cont"]['value'].shape,
                 (data_dim[0], coord+2))
    assert_equal(model_trace.nodes["obs"]['value'].shape,
                 (data_dim[0], torch.prod(tt(data_dim[1:])).item()))


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_jtrvae_disc_sites_dims(data_dim, coord):
    x = torch.randn(*data_dim)
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.jtrVAE(data_dim[1:], 2, 3, coord=coord)
    guide_trace, model_trace = get_enum_traces(model, x)
    assert_equal(model_trace.nodes["latent_disc"]['value'].shape,
                 (3, data_dim[0], 3))
    assert_equal(guide_trace.nodes["latent_disc"]['value'].shape,
                 (3, data_dim[0], 3))


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_jtrvae_cont_sites_fn(data_dim, coord):
    x = torch.randn(*data_dim)
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.jtrVAE(data_dim[1:], 2, 3, coord=coord)
    guide_trace, model_trace = get_enum_traces(model, x)
    assert_(isinstance(model_trace.nodes["latent_cont"]['fn'].base_dist, dist.Normal))
    assert_(isinstance(guide_trace.nodes["latent_cont"]['fn'].base_dist, dist.Normal))
    assert_(isinstance(model_trace.nodes["obs"]['fn'].base_dist, dist.Bernoulli))


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_jtrvae_disc_sites_fn(data_dim, coord):
    x = torch.randn(*data_dim)
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.jtrVAE(data_dim[1:], 2, 3, coord=coord)
    guide_trace, model_trace = get_enum_traces(model, x)
    assert_(isinstance(model_trace.nodes["latent_disc"]['fn'], dist.OneHotCategorical))
    assert_(isinstance(guide_trace.nodes["latent_disc"]['fn'], dist.OneHotCategorical))
