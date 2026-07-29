import torch

from audiolm_pytorch.soundstream import (
    CausalLinearUpsampleConv1d,
    DecoderBlock,
    stream_module,
)


def stream_layer_one_frame_at_a_time(layer, x):
    state = None
    pieces = []
    for frame in x.split(1, dim=-1):
        output, state = layer.forward_stream(frame, state)
        pieces.append(output)
    return torch.cat(pieces, dim=-1)


def stream_module_one_frame_at_a_time(module, x):
    state = None
    pieces = []
    for frame in x.split(1, dim=-1):
        output, state = stream_module(module, frame, state)
        pieces.append(output)
    return torch.cat(pieces, dim=-1)


def test_causal_interpolation_matches_stateful_streaming():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7)

    for interpolation_mode in ("linear", "cubic"):
        layer = CausalLinearUpsampleConv1d(
            3,
            5,
            kernel_size=8,
            stride=4,
            pad_mode="constant",
            interpolation_mode=interpolation_mode,
        ).eval()

        offline = layer(x)
        streamed = stream_layer_one_frame_at_a_time(layer, x)

        assert offline.shape == (2, 5, 28)
        torch.testing.assert_close(
            streamed,
            offline,
            atol=1e-6,
            rtol=1e-5,
        )


def test_cubic_interpolation_reaches_every_latent_knot():
    layer = CausalLinearUpsampleConv1d(
        1,
        1,
        kernel_size=1,
        stride=4,
        pad_mode="constant",
        interpolation_mode="cubic",
    )
    latent = torch.tensor([[[0.0, 1.0, 0.5, 1.5]]])

    upsampled = layer._upsample(latent).reshape(1, 1, 4, 4)

    torch.testing.assert_close(upsampled[..., -1], latent)


def test_split_x8_decoder_block_preserves_shape_and_streaming_equivalence():
    torch.manual_seed(1)
    block = DecoderBlock(
        16,
        8,
        8,
        upsample_mode="linear",
        pad_mode="constant",
        linear_upsample_kernel_min=4,
        interpolation_mode="cubic",
        split_upsample=True,
    ).eval()
    latent = torch.randn(2, 16, 6)

    offline = block(latent)
    streamed = stream_module_one_frame_at_a_time(block, latent)

    assert offline.shape == (2, 8, 48)
    torch.testing.assert_close(
        streamed,
        offline,
        atol=2e-6,
        rtol=2e-5,
    )

