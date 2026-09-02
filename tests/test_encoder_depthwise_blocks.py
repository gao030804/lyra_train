import torch
from torch import nn

from audiolm_pytorch.soundstream import (
    CausalConv1d,
    DepthwiseSeparableCausalConv1d,
    SoundStream,
)


def build_small_codec():
    return SoundStream(
        channels=16,
        channel_mults=(2, 4, 8, 16),
        strides=(2, 4, 5, 8),
        codebook_dim=64,
        codebook_size=256,
        rq_num_quantizers=8,
        use_local_attn=False,
        encoder_depthwise_separable_blocks=(2, 3),
        encoder_depthwise_separable_revision=2,
    )


def test_only_encoder_blocks_3_and_4_use_depthwise_separable_convs():
    model = build_small_codec()
    assert model.encoder_depthwise_separable_revision == 2
    block1, block2, block3, block4 = model.encoder[1:5]

    assert isinstance(block1[0].fn[0], CausalConv1d)
    assert isinstance(block2[0].fn[0], CausalConv1d)

    for block in (block3, block4):
        for residual in block[:3]:
            assert isinstance(
                residual.fn[0],
                DepthwiseSeparableCausalConv1d,
            )
            assert len(residual.fn[0].net) == 2
            assert isinstance(residual.fn[0].net[1], CausalConv1d)
            assert isinstance(residual.fn[1], nn.ReLU)
            assert isinstance(residual.fn[3], nn.ReLU)
        assert isinstance(block[3], DepthwiseSeparableCausalConv1d)
        assert len(block[3].net) == 2
        assert isinstance(block[3].net[1], CausalConv1d)


def test_block3_and_block4_kernel_shapes_and_encoder_output_shape():
    model = build_small_codec()
    block3, block4 = model.encoder[3], model.encoder[4]

    block3_residual_dw = block3[0].fn[0].net[0].conv
    block3_residual_pw = block3[0].fn[0].net[1].conv
    block3_down_dw = block3[3].net[0].conv
    block3_down_pw = block3[3].net[1].conv
    assert tuple(block3_residual_dw.weight.shape) == (64, 1, 7)
    assert block3_residual_dw.groups == 64
    assert tuple(block3_residual_pw.weight.shape) == (64, 64, 1)
    assert tuple(block3_down_dw.weight.shape) == (64, 1, 10)
    assert block3_down_dw.groups == 64
    assert tuple(block3_down_pw.weight.shape) == (128, 64, 1)

    block4_residual_dw = block4[0].fn[0].net[0].conv
    block4_residual_pw = block4[0].fn[0].net[1].conv
    block4_down_dw = block4[3].net[0].conv
    block4_down_pw = block4[3].net[1].conv
    assert tuple(block4_residual_dw.weight.shape) == (128, 1, 7)
    assert block4_residual_dw.groups == 128
    assert tuple(block4_residual_pw.weight.shape) == (128, 128, 1)
    assert tuple(block4_down_dw.weight.shape) == (128, 1, 16)
    assert block4_down_dw.groups == 128
    assert tuple(block4_down_pw.weight.shape) == (256, 128, 1)

    encoded = model.encoder(torch.randn(1, 1, 3200))
    assert encoded.shape == (1, 64, 10)
