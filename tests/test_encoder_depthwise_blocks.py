import torch
from torch import nn

from audiolm_pytorch.soundstream import (
    CausalConv1d,
    DepthwiseSeparableCausalConv1d,
    LowRankPointwiseCausalConv1d,
    SoundStream,
)


def build_small_codec():
    return SoundStream(
        channels=16,
        channel_mults=(2, 4, 8, 16),
        strides=(2, 4, 5, 8),
        codebook_dim=64,
        codebook_size=16,
        rq_num_quantizers=16,
        use_local_attn=False,
        encoder_depthwise_separable_blocks=(1, 2, 3),
        encoder_depthwise_separable_revision=3,
        encoder_low_rank_pointwise_ranks=(
            (0, 0), (8, 16), (16, 32), (32, 64)
        ),
    )


def test_encoder_blocks_2_3_and_4_use_depthwise_separable_convs():
    model = build_small_codec()
    assert model.encoder_depthwise_separable_revision == 3
    block1, block2, block3, block4 = model.encoder[1:5]

    assert isinstance(block1[0].fn[0], CausalConv1d)

    for block in (block2, block3, block4):
        for residual in block[:3]:
            assert isinstance(
                residual.fn[0],
                DepthwiseSeparableCausalConv1d,
            )
            assert isinstance(
                residual.fn[0].net[1],
                LowRankPointwiseCausalConv1d,
            )
            assert isinstance(
                residual.fn[2],
                LowRankPointwiseCausalConv1d,
            )
            assert isinstance(residual.fn[1], nn.ReLU)
            assert isinstance(residual.fn[3], nn.ReLU)
        assert isinstance(block[3], DepthwiseSeparableCausalConv1d)
        assert isinstance(block[3].net[1], LowRankPointwiseCausalConv1d)


def test_block2_block3_and_block4_kernel_shapes_and_encoder_output_shape():
    model = build_small_codec()
    block2, block3, block4 = model.encoder[2], model.encoder[3], model.encoder[4]

    block2_residual_dw = block2[0].fn[0].net[0].conv
    block2_residual_pw = block2[0].fn[0].net[1].net
    block2_down_dw = block2[3].net[0].conv
    block2_down_pw = block2[3].net[1].net
    assert tuple(block2_residual_dw.weight.shape) == (32, 1, 7)
    assert block2_residual_dw.groups == 32
    assert tuple(block2_residual_pw[0].conv.weight.shape) == (8, 32, 1)
    assert tuple(block2_residual_pw[1].conv.weight.shape) == (32, 8, 1)
    assert tuple(block2_down_dw.weight.shape) == (32, 1, 8)
    assert block2_down_dw.groups == 32
    assert tuple(block2_down_pw[0].conv.weight.shape) == (16, 32, 1)
    assert tuple(block2_down_pw[1].conv.weight.shape) == (64, 16, 1)

    block3_residual_dw = block3[0].fn[0].net[0].conv
    block3_residual_pw = block3[0].fn[0].net[1].net
    block3_down_dw = block3[3].net[0].conv
    block3_down_pw = block3[3].net[1].net
    assert tuple(block3_residual_dw.weight.shape) == (64, 1, 7)
    assert block3_residual_dw.groups == 64
    assert tuple(block3_residual_pw[0].conv.weight.shape) == (16, 64, 1)
    assert tuple(block3_residual_pw[1].conv.weight.shape) == (64, 16, 1)
    assert tuple(block3_down_dw.weight.shape) == (64, 1, 10)
    assert block3_down_dw.groups == 64
    assert tuple(block3_down_pw[0].conv.weight.shape) == (32, 64, 1)
    assert tuple(block3_down_pw[1].conv.weight.shape) == (128, 32, 1)

    block4_residual_dw = block4[0].fn[0].net[0].conv
    block4_residual_pw = block4[0].fn[0].net[1].net
    block4_down_dw = block4[3].net[0].conv
    block4_down_pw = block4[3].net[1].net
    assert tuple(block4_residual_dw.weight.shape) == (128, 1, 7)
    assert block4_residual_dw.groups == 128
    assert tuple(block4_residual_pw[0].conv.weight.shape) == (32, 128, 1)
    assert tuple(block4_residual_pw[1].conv.weight.shape) == (128, 32, 1)
    assert tuple(block4_down_dw.weight.shape) == (128, 1, 16)
    assert block4_down_dw.groups == 128
    assert tuple(block4_down_pw[0].conv.weight.shape) == (64, 128, 1)
    assert tuple(block4_down_pw[1].conv.weight.shape) == (256, 64, 1)

    encoded = model.encoder(torch.randn(1, 1, 3200))
    assert encoded.shape == (1, 64, 10)
