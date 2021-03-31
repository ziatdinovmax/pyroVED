import sys
from copy import deepcopy as dc

import torch
import torch.tensor as tt
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
import pyro.infer as infer
from pyro.poutine.enum_messenger import EnumMessenger
import pytest
from numpy.testing import assert_equal, assert_
from numpy import array_equal

sys.path.append("../../")

from pyroved import models, nets, utils


def get_traces(model, *args):
    guide_trace = pyro.poutine.trace(model.guide).get_trace(*args)
    model_trace = pyro.poutine.trace(
        pyro.poutine.replay(model.model, trace=guide_trace)).get_trace(*args)
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


def assert_weights_equal(m1, m2):
    eq_w = []
    for p1, p2 in zip(m1.values(), m2.values()):
        eq_w.append(array_equal(
            p1.detach().cpu().numpy(),
            p2.detach().cpu().numpy()))
    return all(eq_w)


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8), (3, 64)])
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
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8), (3, 64)])
def test_trvae_sites_fn(data_dim, coord):
    x = torch.randn(*data_dim)
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.trVAE(data_dim[1:], coord=coord)
    guide_trace, model_trace = get_traces(model, x)
    assert_(isinstance(model_trace.nodes["latent"]['fn'].base_dist, dist.Normal))
    assert_(isinstance(guide_trace.nodes["latent"]['fn'].base_dist, dist.Normal))
    assert_(isinstance(model_trace.nodes["obs"]['fn'].base_dist, dist.Bernoulli))


@pytest.mark.parametrize("input_dim, output_dim",
                         [((8,), (8, 8)), ((8, 8), (8,)),
                          ((8,), (8,)), ((8, 8), (8, 8))])
def test_ved_sites_dims(input_dim, output_dim):
    x = torch.randn(2, 1, *input_dim)
    y = torch.randn(2, 1, *output_dim)
    model = models.VED(input_dim, output_dim)
    guide_trace, model_trace = get_traces(model, x, y)
    assert_equal(model_trace.nodes["z"]['value'].shape,
                 (x.shape[0], 2))
    assert_equal(guide_trace.nodes["z"]['value'].shape,
                 (x.shape[0], 2))
    assert_equal(model_trace.nodes["obs"]['value'].shape,
                 (y.shape[0], torch.prod(tt(output_dim)).item()))


@pytest.mark.parametrize("input_dim, output_dim",
                         [((8,), (8, 8)), ((8, 8), (8,)),
                          ((8,), (8,)), ((8, 8), (8, 8))])
def test_ved_sites_fn(input_dim, output_dim):
    x = torch.randn(2, 1, *input_dim)
    y = torch.randn(2, 1, *output_dim)
    model = models.VED(input_dim, output_dim)
    guide_trace, model_trace = get_traces(model, x, y)
    assert_(isinstance(model_trace.nodes["z"]['fn'].base_dist, dist.Normal))
    assert_(isinstance(guide_trace.nodes["z"]['fn'].base_dist, dist.Normal))
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


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_sstrvae_cont_sites_dims(data_dim, coord):
    x = torch.randn(data_dim[0], torch.prod(tt(data_dim[1:])).item())
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.sstrVAE(data_dim[1:], 2, 3, coord=coord)
    guide_trace, model_trace = get_enum_traces(model, x)
    assert_equal(model_trace.nodes["z"]['value'].shape,
                 (3, data_dim[0], coord+2))
    assert_equal(guide_trace.nodes["z"]['value'].shape,
                 (3, data_dim[0], coord+2))
    assert_equal(model_trace.nodes["x"]['value'].shape,
                 (data_dim[0], torch.prod(tt(data_dim[1:])).item()))


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_sstrvae_disc_sites_dims(data_dim, coord):
    x = torch.randn(data_dim[0], torch.prod(tt(data_dim[1:])).item())
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.sstrVAE(data_dim[1:], 2, 3, coord=coord)
    guide_trace, model_trace = get_enum_traces(model, x)
    assert_equal(model_trace.nodes["y"]['value'].shape,
                 (3, data_dim[0], 3))
    assert_equal(guide_trace.nodes["y"]['value'].shape,
                 (3, data_dim[0], 3))


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_sstrvae_cont_sites_fn(data_dim, coord):
    x = torch.randn(data_dim[0], torch.prod(tt(data_dim[1:])).item())
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.sstrVAE(data_dim[1:], 2, 3, coord=coord)
    guide_trace, model_trace = get_enum_traces(model, x)
    assert_(isinstance(model_trace.nodes["z"]['fn'].base_dist, dist.Normal))
    assert_(isinstance(guide_trace.nodes["z"]['fn'].base_dist, dist.Normal))
    assert_(isinstance(model_trace.nodes["x"]['fn'].base_dist, dist.Bernoulli))


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_sstrvae_disc_sites_fn(data_dim, coord):
    x = torch.randn(data_dim[0], torch.prod(tt(data_dim[1:])).item())
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.sstrVAE(data_dim[1:], 2, 3, coord=coord)
    guide_trace, model_trace = get_enum_traces(model, x)
    assert_(isinstance(model_trace.nodes["y"]['fn'], dist.OneHotCategorical))
    assert_(isinstance(guide_trace.nodes["y"]['fn'], dist.OneHotCategorical))


@pytest.mark.parametrize(
    "sampler, expected_dist",
    [("gaussian", dist.Normal), ("bernoulli", dist.Bernoulli),
     ("continuous_bernoulli", dist.ContinuousBernoulli)])
def test_trvae_decoder_sampler(sampler, expected_dist):
    data_dim = (2, 8, 8)
    x = torch.randn(*data_dim)
    model = models.trVAE(data_dim[1:], coord=1, sampler_d=sampler)
    _, model_trace = get_traces(model, x)
    assert_(isinstance(model_trace.nodes["obs"]['fn'].base_dist, expected_dist))


@pytest.mark.parametrize(
    "sampler, expected_dist",
    [("gaussian", dist.Normal), ("bernoulli", dist.Bernoulli),
     ("continuous_bernoulli", dist.ContinuousBernoulli)])
def test_ved_decoder_sampler(sampler, expected_dist):
    input_dim = (8, 8)
    output_dim = (8,)
    x = torch.randn(2, 1, *input_dim)
    y = torch.randn(2, 1, *output_dim)
    model = models.VED(input_dim, output_dim, sampler_d=sampler)
    _, model_trace = get_traces(model, x, y)
    assert_(isinstance(model_trace.nodes["obs"]['fn'].base_dist, expected_dist))


@pytest.mark.parametrize(
    "sampler, expected_dist",
    [("gaussian", dist.Normal), ("bernoulli", dist.Bernoulli),
     ("continuous_bernoulli", dist.ContinuousBernoulli)])
def test_jtrvae_decoder_sampler(sampler, expected_dist):
    data_dim = (2, 8, 8)
    x = torch.randn(*data_dim)
    model = models.jtrVAE(data_dim[1:], 2, 3, coord=1, sampler_d=sampler)
    _, model_trace = get_enum_traces(model, x)
    assert_(isinstance(model_trace.nodes["obs"]['fn'].base_dist, expected_dist))


@pytest.mark.parametrize(
    "sampler, expected_dist",
    [("gaussian", dist.Normal), ("bernoulli", dist.Bernoulli),
     ("continuous_bernoulli", dist.ContinuousBernoulli)])
def test_sstrvae_decoder_sampler(sampler, expected_dist):
    data_dim = (2, 64)
    x = torch.randn(*data_dim)
    model = models.sstrVAE(data_dim[1:], 2, 3, coord=1, sampler_d=sampler)
    _, model_trace = get_enum_traces(model, x)
    assert_(isinstance(model_trace.nodes["x"]['fn'].base_dist, expected_dist))


@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_basevae_encode_x(data_dim):
    x = torch.randn(*data_dim)
    vae = models.base.baseVAE()
    encoder_net = nets.fcEncoderNet(data_dim[1:], 2, 0)
    vae.set_encoder(encoder_net)
    encoded = vae._encode(x)
    assert_equal(encoded[:, :2].shape, (data_dim[0], 2))
    assert_equal(encoded[:, 2:].shape, (data_dim[0], 2))


def test_basevae_encode_xy():
    data_dim = (2, 64)
    x = torch.randn(*data_dim)
    alpha = torch.ones(data_dim[0], 3) / 3
    y = dist.OneHotCategorical(alpha).sample()
    vae = models.base.baseVAE()
    encoder_net = nets.fcEncoderNet(data_dim[1:], 2, 3)
    vae.set_encoder(encoder_net)
    encoded = vae._encode(x, y)
    assert_equal(encoded[:, :2].shape, (data_dim[0], 2))
    assert_equal(encoded[:, 2:].shape, (data_dim[0], 2))


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_basevae_decode_x(data_dim, coord):
    z = torch.randn(data_dim[0], 2)
    vae = models.base.baseVAE()
    vae.coord = coord
    vae.grid = utils.generate_grid(data_dim[1:]).to(vae.device)
    dnet = nets.sDecoderNet if coord in [1, 2, 3] else nets.fcDecoderNet
    decoder_net = dnet(data_dim[1:], 2)
    vae.set_decoder(decoder_net)
    decoded = vae._decode(z)
    assert_equal(decoded.squeeze().shape, data_dim)


@pytest.mark.parametrize("vae_model", [models.jtrVAE, models.sstrVAE])
@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(8,), (8, 8), (8,), (8, 8)])
def test_jsstrvae_decode(vae_model, data_dim, coord):
    if coord > 0:
        coord = coord if len(data_dim) > 1 else 1
    model = vae_model(data_dim, 2, 3, coord=coord)
    z_coord = torch.tensor([0.0, 0.0]).unsqueeze(0)
    y = utils.to_onehot(torch.tensor(0).unsqueeze(0), 3)
    decoded = model.decode(z_coord, y)
    assert_equal(decoded.squeeze().shape, data_dim)


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(8,), (8, 8)])
def test_trvae_decode(data_dim, coord):
    if coord > 0:
        coord = coord if len(data_dim) > 1 else 1
    model = models.trVAE(data_dim, coord=coord)
    z_coord = torch.tensor([0.0, 0.0]).unsqueeze(0)
    decoded = model.decode(z_coord)
    assert_equal(decoded.squeeze().shape, data_dim)


@pytest.mark.parametrize("input_dim, output_dim",
                         [((8,), (8, 8)), ((8, 8), (8,)),
                          ((8,), (8,)), ((8, 8), (8, 8))])
def test_ved_decode(input_dim, output_dim):
    z_coord = torch.tensor([0.0, 0.0]).unsqueeze(0)
    model = models.VED(input_dim, output_dim)
    decoded = model.decode(z_coord)
    assert_equal(decoded.squeeze().shape, output_dim)


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(8,), (8, 8)])
def test_ctrvae_decode(data_dim, coord):
    if coord > 0:
        coord = coord if len(data_dim) > 1 else 1
    model = models.trVAE(data_dim, num_classes=3, coord=coord)
    z_coord = torch.tensor([0.0, 0.0]).unsqueeze(0)
    y = utils.to_onehot(torch.tensor(0).unsqueeze(0), 3)
    decoded = model.decode(z_coord, y)
    assert_equal(decoded.squeeze().shape, data_dim)


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_trvae_encode(data_dim, coord):
    x = torch.randn(*data_dim)
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.trVAE(data_dim[1:], 2, coord=coord)
    encoded = model.encode(x)
    assert_equal(encoded[0].shape, (data_dim[0], coord+2))
    assert_equal(encoded[0].shape, encoded[1].shape)


@pytest.mark.parametrize("input_dim, output_dim",
                         [((8,), (8, 8)), ((8, 8), (8,)),
                          ((8,), (8,)), ((8, 8), (8, 8))])
def test_ved_encode(input_dim, output_dim):
    x = torch.randn(2, 1, *input_dim)
    model = models.VED(input_dim, output_dim)
    encoded = model.encode(x)
    assert_equal(encoded[0].shape, (x.shape[0], 2))
    assert_equal(encoded[0].shape, encoded[1].shape)


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_jtrvae_encode(data_dim, coord):
    x = torch.randn(*data_dim)
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.jtrVAE(data_dim[1:], 2, 3, coord=coord)
    encoded = model.encode(x)
    assert_equal(encoded[0].shape, encoded[1].shape)
    assert_equal(encoded[0].shape, (data_dim[0], coord+2))
    assert_equal(encoded[2].shape, (data_dim[0],))


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(2, 8), (2, 8, 8), (3, 8), (3, 8, 8)])
def test_sstrvae_encode(data_dim, coord):
    x = torch.randn(data_dim[0], torch.prod(tt(data_dim[1:])).item())
    if coord > 0:
        coord = coord if len(data_dim[1:]) > 1 else 1
    model = models.sstrVAE(data_dim[1:], 2, 5, coord=coord)
    encoded = model.encode(x)
    assert_equal(encoded[0].shape, encoded[1].shape)
    assert_equal(encoded[0].shape, (data_dim[0], coord+2))
    assert_equal(encoded[2].shape, (data_dim[0],))


@pytest.mark.parametrize("num_classes", [0, 2, 3])
@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(8,), (8, 8)])
def test_trvae_manifold2d(data_dim, coord, num_classes):
    if coord > 0:
        coord = coord if len(data_dim) > 1 else 1
    model = models.trVAE(data_dim, num_classes=num_classes, coord=coord)
    decoded_grid = model.manifold2d(4, plot=False)
    assert_equal(decoded_grid.squeeze().shape, (16, *data_dim))


@pytest.mark.parametrize("input_dim, output_dim",
                         [((8,), (8, 8)), ((8, 8), (8,)),
                          ((8,), (8,)), ((8, 8), (8, 8))])
def test_ved_manifold2d(input_dim, output_dim):
    model = models.VED(input_dim, output_dim)
    decoded_grid = model.manifold2d(4, plot=False)
    assert_equal(decoded_grid.squeeze().shape, (16, *output_dim))


@pytest.mark.parametrize("vae_model", [models.jtrVAE, models.sstrVAE])
@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(8,), (8, 8)])
def test_jsstrvae_manifold2d(vae_model, data_dim, coord):
    if coord > 0:
        coord = coord if len(data_dim) > 1 else 1
    model = vae_model(data_dim, 2, 3, coord=coord)
    decoded_grid = model.manifold2d(4, plot=False)
    assert_equal(decoded_grid.squeeze().shape, (16, *data_dim))


@pytest.fixture(scope='session')
@pytest.mark.parametrize("coord", [0, 1, 2, 3])
def test_save_load_basevae(coord):
    data_dim = (5, 8, 8)
    vae = models.base.baseVAE()
    encoder_net = nets.fcEncoderNet(data_dim[1:], 2+coord, 0)
    dnet = nets.sDecoderNet if coord in [1, 2, 3] else nets.fcDecoderNet
    decoder_net = dnet(data_dim, 2, 0)
    vae.set_encoder(encoder_net)
    vae.set_decoder(decoder_net)
    weights_init = dc(vae.state_dict())
    vae.save_weights("my_weights")
    vae.load_weights("my_weights.pt")
    weights_loaded = vae.state_dict()
    assert_(assert_weights_equal(weights_loaded, weights_init))