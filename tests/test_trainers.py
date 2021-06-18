import sys
from copy import deepcopy as dc

import torch
import pyro.distributions as dist
import pytest
import numpy as np
from numpy.testing import assert_

sys.path.append("../../")

from pyroved import models, utils, trainers

tt = torch.tensor


def assert_weights_equal(m1, m2):
    eq_w = []
    for p1, p2 in zip(m1.values(), m2.values()):
        eq_w.append(np.array_equal(
            p1.detach().cpu().numpy(),
            p2.detach().cpu().numpy()))
    return all(eq_w)


@pytest.mark.parametrize("invariances", [None, ['r'], ['s'], ['t'], ['r', 't', 's']])
def test_svi_trainer_trvae(invariances):
    data_dim = (5, 8, 8)
    train_data = torch.randn(*data_dim)
    test_data = torch.randn(*data_dim)
    train_loader = utils.init_dataloader(train_data, batch_size=2)
    test_loader = utils.init_dataloader(test_data, batch_size=2)
    vae = models.iVAE(data_dim[1:], 2, invariances)
    trainer = trainers.SVItrainer(vae)
    weights_before = dc(vae.state_dict())
    for _ in range(2):
        trainer.step(train_loader, test_loader)
    weights_after = vae.state_dict()
    assert_(not torch.isnan(tt(trainer.loss_history["training_loss"])).any())
    assert_(not assert_weights_equal(weights_before, weights_after))


@pytest.mark.parametrize("invariances", [None, ['r'], ['s'], ['t'], ['r', 't', 's']])
def test_svi_trainer_jtrvae(invariances):
    data_dim = (6, 8, 8)
    train_data = torch.randn(*data_dim)
    train_loader = utils.init_dataloader(train_data, batch_size=2)
    vae = models.jiVAE(data_dim[1:], 2, 3, invariances)
    trainer = trainers.SVItrainer(vae, enumerate_parallel=True)
    weights_before = dc(vae.state_dict())
    for _ in range(2):
        trainer.step(train_loader)
    weights_after = vae.state_dict()
    assert_(not torch.isnan(tt(trainer.loss_history["training_loss"])).any())
    assert_(not assert_weights_equal(weights_before, weights_after))


@pytest.mark.parametrize("invariances", [None, ['r'], ['s'], ['t'], ['r', 't', 's']])
def test_auxsvi_trainer_cls(invariances):
    data_dim = (5, 8, 8)
    train_unsup = torch.randn(data_dim[0], torch.prod(tt(data_dim[1:])).item())
    train_sup = train_unsup + .1 * torch.randn_like(train_unsup)
    labels = dist.OneHotCategorical(torch.ones(data_dim[0], 3)).sample()
    loader_unsup, loader_sup, loader_val = utils.init_ssvae_dataloaders(
        train_unsup, (train_sup, labels), (train_sup, labels), batch_size=2)
    vae = models.ssiVAE(data_dim[1:], 2, 3, invariances)
    trainer = trainers.auxSVItrainer(vae)
    weights_before = dc(vae.state_dict())
    for _ in range(2):
        trainer.step(loader_unsup, loader_sup, loader_val)
    weights_after = vae.state_dict()
    assert_(not torch.isnan(tt(trainer.history["training_loss"])).any())
    assert_(not assert_weights_equal(weights_before, weights_after))


@pytest.mark.parametrize("c_dim", [1, 2])
@pytest.mark.parametrize("invariances", [None, ['r'], ['s'], ['t'], ['r', 't', 's']])
def test_auxsvi_trainer_reg(c_dim, invariances):
    data_dim = (5, 8, 8)
    train_unsup = torch.randn(data_dim[0], torch.prod(tt(data_dim[1:])).item())
    train_sup = train_unsup + .1 * torch.randn_like(train_unsup)
    gt = torch.randn(data_dim[0], c_dim)

    loader_unsup, loader_sup, loader_val = utils.init_ssvae_dataloaders(
        train_unsup, (train_sup, gt), (train_sup, gt), batch_size=2)
    vae = models.ss_reg_iVAE(data_dim[1:], 2, c_dim, invariances)
    trainer = trainers.auxSVItrainer(vae, task="regression")
    weights_before = dc(vae.state_dict())
    for _ in range(2):
        trainer.step(loader_unsup, loader_sup, loader_val)
    weights_after = vae.state_dict()
    assert_(not torch.isnan(tt(trainer.history["training_loss"])).any())
    assert_(not assert_weights_equal(weights_before, weights_after))


@pytest.mark.parametrize("invariances", [None, ['r'], ['s'], ['t'], ['r', 't', 's']])
def test_auxsvi_trainer_swa(invariances):
    data_dim = (5, 8, 8)
    train_unsup = torch.randn(data_dim[0], torch.prod(tt(data_dim[1:])).item())
    train_sup = train_unsup + .1 * torch.randn_like(train_unsup)
    labels = dist.OneHotCategorical(torch.ones(data_dim[0], 3)).sample()
    loader_unsup, loader_sup, _ = utils.init_ssvae_dataloaders(
        train_unsup, (train_sup, labels), (train_sup, labels), batch_size=2)
    vae = models.ssiVAE(data_dim[1:], 2, 3, invariances)
    trainer = trainers.auxSVItrainer(vae)
    for _ in range(3):
        trainer.step(loader_unsup, loader_sup)
        trainer.save_running_weights("encoder_y")
    weights_final = dc(vae.encoder_y.state_dict())
    trainer.average_weights("encoder_y")
    weights_aver = vae.encoder_y.state_dict()
    assert_(not assert_weights_equal(weights_final, weights_aver))


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
