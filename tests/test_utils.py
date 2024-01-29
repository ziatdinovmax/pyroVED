import pytest
import torch
import pyro
import pyro.contrib.gp as gp
from pyroved.utils import gp_model 
from pyroved import models

def test_gp_model_output_shape():
    input_dim = 3
    num_samples = 5
    encoded_X = torch.randn(num_samples, input_dim)  # Random tensor for encoded_X
    y = torch.randn(num_samples)  # Random tensor for y
    gpr = gp_model(input_dim, encoded_X, y)
    with torch.no_grad():
        predictions, _ = gpr(encoded_X)
    assert predictions.shape == y.shape, "Output tensor shape mismatch"
