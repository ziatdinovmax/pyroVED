import sys
import pytest
from numpy.testing import assert_equal
import torch
import torch.nn as nn

sys.path.append("../../../")

from pyroved.nets import conv


@pytest.mark.parametrize("hidden_dim, bnorm, nbnorm",
                         [([(8,)], True, 1), ([(8,)], False, 0),
                          ([(8,), (16, 16)], True, 3), ([(8,), (16, 16)], False, 0)])
def test_feature_extractor_bnorm(hidden_dim, bnorm, nbnorm):
    c = conv.FeatureExtractor(2, conv_filters=hidden_dim, batchnorm=bnorm)
    nbnorm_ = len([k for k in c.state_dict().keys() if 'running_mean' in k])
    assert_equal(nbnorm, nbnorm_)


@pytest.mark.parametrize("activation, activation_expected",
                         [("relu", nn.modules.activation.ReLU),
                          ("lrelu", nn.modules.activation.LeakyReLU),
                          ("softplus", nn.modules.activation.Softplus),
                          ("tanh", nn.modules.activation.Tanh)])
def test_feature_extractor_activation(activation, activation_expected):
    conv_ = conv.FeatureExtractor(2, conv_filters=[(8, 8)], activation=activation)
    activations_cnt = 0
    for c1 in conv_.children():
        for c2 in c1.children():
            if isinstance(c2, activation_expected):
                activations_cnt += 1
    assert_equal(activations_cnt, 2)


@pytest.mark.parametrize("dim, conv_expected",
                         [(1, nn.modules.Conv1d),
                          (2, nn.modules.Conv2d),
                          (3, nn.modules.Conv3d)])
def test_feature_extractor_dim(dim, conv_expected):
    conv_ = conv.FeatureExtractor(dim, conv_filters=[(8, 8)])
    conv_cnt = 0
    for c1 in conv_.children():
        for c2 in c1.children():
            if isinstance(c2, conv_expected):
                conv_cnt += 1
    assert_equal(conv_cnt, 2)


@pytest.mark.parametrize("dim, size", [(1, [8]), (2, [8, 8]), (3, [8, 8, 8])])
def test_feature_extractor_last_pool(dim, size):
    data = torch.randn(2, 1, *size)
    conv_ = conv.FeatureExtractor(dim, conv_filters=[(8, 8)], pool_last=True)
    out = conv_(data)
    size_ = sum([out.size(i+2) for i in range(dim)])
    assert_equal(size_, sum(size) / 2)


@pytest.mark.parametrize("dim, size", [(1, [8]), (2, [8, 8]), (3, [8, 8, 8])])
def test_feature_extractor_no_last_pool(dim, size):
    data = torch.randn(2, 1, *size)
    conv_ = conv.FeatureExtractor(dim, conv_filters=[(8, 8)], pool_last=False)
    out = conv_(data)
    size_ = sum([out.size(i+2) for i in range(dim)])
    assert_equal(size_, sum(size))


@pytest.mark.parametrize("dim, size", [(1, [8]), (2, [8, 8]), (3, [8, 8, 8])])
def test_upsample_block(dim, size):
    data = torch.randn(2, 2, *size)
    up = conv.UpsampleBlock(dim, 2, 2, mode="nearest")
    out = up(data)
    size_ = sum([out.size(i+2) for i in range(dim)])
    assert_equal(size_, sum(size) * 2)


@pytest.mark.parametrize("in_channels, out_channels",
                         [(8, 8), (8, 4), (4, 8)])
def test_upsampleblock_change_number_of_channels(in_channels, out_channels):
    data = torch.randn(4, in_channels, 8, 8)
    up = conv.UpsampleBlock(2, in_channels, out_channels)
    out = up(data)
    assert_equal(out.size(1), out_channels)


@pytest.mark.parametrize("latent_dim", [1, 2, 5])
@pytest.mark.parametrize("input_channels", [1, 2, 3])
@pytest.mark.parametrize("input_dim", [(8,), (8, 8), (8, 8, 8)])
def test_conv_encoder_output(input_dim, input_channels, latent_dim):
    x = torch.randn(5, input_channels, *input_dim)
    encoder = conv.convEncoderNet(
        input_dim, latent_dim, input_channels, hidden_dim=[(8,), (8, 8)])
    z1, z2 = encoder(x)
    assert_equal(z1.shape, z2.shape)
    assert_equal(z1.shape, (x.shape[0], latent_dim))


@pytest.mark.parametrize("latent_dim", [1, 2, 5])
@pytest.mark.parametrize("output_channels", [1, 2, 3])
@pytest.mark.parametrize("output_dim", [(8,), (8, 8), (8, 8, 8)])
def test_conv_decoder_output(latent_dim, output_dim, output_channels):
    z = torch.randn(5, latent_dim)
    decoder = conv.convDecoderNet(
        latent_dim, output_dim, output_channels, hidden_dim=[(8, 8), (8,)])
    x = decoder(z)
    assert_equal(x.shape, (z.shape[0], output_channels, *output_dim))
