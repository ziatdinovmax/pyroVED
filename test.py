import sys
from copy import deepcopy as dc

import torch
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
import pyro.infer as infer
from pyro.poutine.enum_messenger import EnumMessenger
import pytest
from numpy.testing import assert_equal, assert_
from numpy import array_equal


from pyroved import models, nets, utils

tt = torch.tensor


def get_traces(model, *args):
    guide_trace = pyro.poutine.trace(model.guide).get_trace(*args)
    model_trace = pyro.poutine.trace(
        pyro.poutine.replay(model.model, trace=guide_trace)).get_trace(*args)
    return guide_trace, model_trace

data_dim = (3, 8, 8)
x = torch.randn(data_dim[0], torch.prod(tt(data_dim[1:])).item())
model = models.ss_reg_iVAE(data_dim[1:], 2, 3, invariances=None)
guide_trace, model_trace = get_traces(model, x)
print(model_trace.nodes["y"]['fn'])
