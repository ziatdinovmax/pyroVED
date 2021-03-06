import sys
from copy import deepcopy as dc

import torch
import torch.tensor as tt
import pyro.distributions as dist
import pytest
import numpy as np
from numpy.testing import assert_

sys.path.append("../../")

from pyroved import models, utils, trainers


def assert_weights_equal(m1, m2):
    eq_w = []
    for p1, p2 in zip(m1.values(), m2.values()):
        eq_w.append(np.array_equal(
            p1.detach().cpu().numpy(),
            p2.detach().cpu().numpy()))
    return all(eq_w)


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(5, 8), (5, 8, 8), (6, 8), (6, 8, 8), (6, 64)])
def test_svi_trainer_trvae(data_dim, coord):
    train_data = torch.randn(*data_dim)
    train_loader = utils.init_dataloader(train_data, batch_size=2)
    vae = models.trVAE(data_dim[1:], 2, coord=coord)
    trainer = trainers.SVItrainer(vae)
    weights_before = dc(vae.state_dict())
    for _ in range(2):
        trainer.step(train_loader)
    weights_after = vae.state_dict()
    assert_(not torch.isnan(tt(trainer.loss_history["training_loss"])).any())
    assert_(not assert_weights_equal(weights_before, weights_after))


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(5, 8), (5, 8, 8), (6, 8), (6, 8, 8), (6, 64)])
def test_svi_trainer_jtrvae(data_dim, coord):
    train_data = torch.randn(*data_dim)
    train_loader = utils.init_dataloader(train_data, batch_size=2)
    vae = models.jtrVAE(data_dim[1:], 2, 3, coord=coord)
    trainer = trainers.SVItrainer(vae, enumerate_parallel=True)
    weights_before = dc(vae.state_dict())
    for _ in range(2):
        trainer.step(train_loader)
    weights_after = vae.state_dict()
    assert_(not torch.isnan(tt(trainer.loss_history["training_loss"])).any())
    assert_(not assert_weights_equal(weights_before, weights_after))


@pytest.mark.parametrize("coord", [0, 1, 2, 3])
@pytest.mark.parametrize("data_dim", [(5, 8), (5, 8, 8), (6, 8), (6, 8, 8)])
def test_auxsvi_trainer(data_dim, coord):
    train_unsup = torch.randn(data_dim[0], torch.prod(tt(data_dim[1:])).item())
    train_sup = train_unsup + .1 * torch.randn_like(train_unsup)
    labels = dist.OneHotCategorical(torch.ones(data_dim[0], 3)).sample()
    loader_unsup, loader_sup, _ = utils.init_ssvae_dataloaders(
        train_unsup, (train_sup, labels), (train_sup, labels), batch_size=2)
    vae = models.sstrVAE(data_dim[1:], 2, 3, coord=coord)
    trainer = trainers.auxSVItrainer(vae)
    weights_before = dc(vae.state_dict())
    for _ in range(2):
        trainer.step(loader_unsup, loader_sup)
    weights_after = vae.state_dict()
    assert_(not torch.isnan(tt(trainer.history["training_loss"])).any())
    assert_(not assert_weights_equal(weights_before, weights_after))


@pytest.mark.parametrize("input_dim, output_dim",
                         [((8,), (8, 8)), ((8, 8), (8,)),
                          ((8,), (8,)), ((8, 8), (8, 8))])
def test_svi_trainer_ved(input_dim, output_dim):
    train_data_x = torch.randn(5, 1, *input_dim)
    train_data_y = torch.randn(5, 1, *output_dim)
    train_loader = utils.init_dataloader(train_data_x, train_data_y, batch_size=2)
    vae = models.VED(input_dim, output_dim)
    trainer = trainers.SVItrainer(vae)
    weights_before = dc(vae.state_dict())
    for _ in range(2):
        trainer.step(train_loader)
    weights_after = vae.state_dict()
    assert_(not torch.isnan(tt(trainer.loss_history["training_loss"])).any())
    assert_(not assert_weights_equal(weights_before, weights_after))
