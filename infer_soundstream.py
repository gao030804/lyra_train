from __future__ import annotations

import argparse
import math
import os
import pickle
import re
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RUNTIME_TMP_DIR = PROJECT_DIR / ".runtime-tmp"
RUNTIME_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMP", str(RUNTIME_TMP_DIR))
os.environ.setdefault("TEMP", str(RUNTIME_TMP_DIR))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
tempfile.tempdir = str(RUNTIME_TMP_DIR)

import torch
import torch.nn.functional as F
import torchaudio

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional runtime fallback
    sf = None

from audiolm_pytorch import FrameStreamingSoundStream, SoundStream


DEFAULT_RESULTS_DIRS = (
    PROJECT_DIR / "results" / "stream-finetune-long-64d-23q",
    PROJECT_DIR / "results" / "stream-finetune-64d-23q",
    PROJECT_DIR / "results" / "gan-pretrain-64d-23q",
    PROJECT_DIR / "results" / "recon-pretrain-saturday-baseline-6gpu",
    PROJECT_DIR / "results" / "recon-pretrain-64d-23q",
    PROJECT_DIR / "results" / "overfit-64d-23q",
    PROJECT_DIR / "results" / "soundstream-3k2-finetune",
    PROJECT_DIR / "results" / "soundstream-3k2-pretrain",
    PROJECT_DIR / "results" / "soundstream-librispeech",
    PROJECT_DIR / "results" / "soundstream-3k2",
)

BITRATE_TO_QUANTIZERS = {
    3200: 8,
    6000: 15,
    9200: 23,
}


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"soundstream\.(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(*, use_ema: bool) -> Path | None:
    best_names = (
        (
            "best_by_aligned_si_sdr.pt",
            "best_selected.pt",
            "best_raw_online_by_aligned_si_sdr.pt",
            "best_ema.pt",
            "best.pt",
        )
        if use_ema
        else (
            "best_by_aligned_si_sdr.pt",
            "best_selected.pt",
            "best.pt",
        )
    )

    for results_dir in DEFAULT_RESULTS_DIRS:
        for best_name in best_names:
            best_checkpoint = results_dir / best_name
            if best_checkpoint.exists():
                return best_checkpoint

    for results_dir in DEFAULT_RESULTS_DIRS:
        if not results_dir.exists():
            continue

        checkpoints = [
            path
            for path in results_dir.glob("soundstream.*.pt")
            if checkpoint_step(path) >= 0
        ]
        if checkpoints:
            return max(checkpoints, key=checkpoint_step)

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SoundStream inference and save reconstructed audio."
    )

    parser.add_argument(
        "input_audio",
        type=Path,
        help="Input wav/flac/mp3/webm audio file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to soundstream.<step>.pt. Defaults to the latest checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "reconstructed.flac",
        help="Output reconstructed audio path.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    parser.add_argument(
        "--weights",
        choices=("auto", "online", "ema"),
        default="auto",
        help=(
            "Checkpoint weight source. 'auto' honors model-only checkpoint "
            "metadata and otherwise defaults ambiguous periodic checkpoints "
            "to online weights."
        ),
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Deprecated alias for --weights online.",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        choices=tuple(BITRATE_TO_QUANTIZERS),
        default=None,
        help="Codec payload bitrate. Defaults to all RVQ quantizers in the checkpoint.",
    )
    parser.add_argument(
        "--block-seconds",
        type=float,
        default=5.0,
        help=(
            "Block size for non-streaming SoundStream inference. "
            "Defaults to the stage-1 final-test setting."
        ),
    )
    parser.add_argument(
        "--context-ms",
        type=float,
        default=60.0,
        help=(
            "Previous-audio context prepended to each non-streaming block. "
            "The context is discarded from the output. Defaults to the stage-1 final-test setting."
        ),
    )
    parser.add_argument(
        "--whole-file",
        action="store_true",
        help="Disable block/context inference and run ordinary whole-file inference.",
    )
    parser.add_argument(
        "--bypass-rvq",
        action="store_true",
        help=(
            "Diagnostic mode: reconstruct with Encoder -> Decoder and skip RVQ. "
            "Use this to determine whether noise comes from RVQ quantization or "
            "from the encoder/decoder itself."
        ),
    )

    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(str(path), map_location="cpu")


def soundstream_from_checkpoint(pkg: dict, use_ema: bool) -> SoundStream:
    if "config" not in pkg:
        raise ValueError("Checkpoint does not contain a SoundStream config.")

    config = pickle.loads(pkg["config"])
    model_cls = (
        FrameStreamingSoundStream
        if "stream_frame_size" in config
        else SoundStream
    )
    model = model_cls(**config)

    if use_ema and "ema_model" in pkg:
        state_dict = {
            key.removeprefix("ema_model."): value
            for key, value in pkg["ema_model"].items()
            if key.startswith("ema_model.")
        }
    else:
        state_dict = pkg["model"]

    # Inference only uses encoder, RVQ and decoder.  Loading generator state
    # strictly keeps older checkpoints usable after discriminator-only
    # architecture corrections, without hiding any generator mismatch.
    model.load_generator_state_dict(state_dict)
    return model


def load_audio(path: Path, target_sample_rate: int) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Input audio not found: {path}")

    if sf is not None and path.suffix.lower() in {".flac", ".wav", ".ogg"}:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        wave = torch.from_numpy(audio.T).contiguous()
    else:
        wave, sample_rate = torchaudio.load(str(path))

    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        wave = torchaudio.functional.resample(
            wave,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )

    return wave


def save_audio(path: Path, wave: torch.Tensor, sample_rate: int) -> None:
    suffix = path.suffix.lower()
    if sf is not None and suffix in {".flac", ".wav", ".ogg"}:
        sf.write(str(path), wave.squeeze(0).detach().cpu().numpy(), sample_rate)
        return

    torchaudio.save(str(path), wave, sample_rate)


def run_frame_streaming_codec(
    model: FrameStreamingSoundStream,
    wave: torch.Tensor,
    num_quantizers: int,
    bypass_rvq: bool = False,
) -> torch.Tensor:
    encoder_state = None
    decoder_state = None
    reconstructed_frames = []
    batched_wave = wave.unsqueeze(0)

    for frame in batched_wave.split(model.stream_frame_size, dim=-1):
        if bypass_rvq:
            latent, encoder_state = model.encode_frame_latent(
                frame,
                state=encoder_state,
            )
            reconstructed, decoder_state = model.decode_frame(
                latent,
                state=decoder_state,
            )
        else:
            _, codes, _, encoder_state = model.encode_frame(
                frame,
                state=encoder_state,
                num_quantizers=num_quantizers,
            )
            reconstructed, decoder_state = model.decode_codes_frame(
                codes,
                state=decoder_state,
            )
        reconstructed_frames.append(reconstructed)

    return torch.cat(reconstructed_frames, dim=-1)


def run_block_context_codec(
    model: SoundStream,
    wave: torch.Tensor,
    *,
    num_quantizers: int,
    block_seconds: float,
    context_ms: float,
    bypass_rvq: bool = False,
) -> torch.Tensor:
    sample_rate = int(model.target_sample_hz)
    frame_multiple = int(model.seq_len_multiple_of)
    block_samples = int(round(block_seconds * sample_rate))
    context_samples = int(round(context_ms * sample_rate / 1000.0))

    if block_samples <= 0:
        raise ValueError("--block-seconds must be greater than 0.")
    if context_samples < 0:
        raise ValueError("--context-ms cannot be negative.")
    if block_samples % frame_multiple != 0:
        raise ValueError(
            f"--block-seconds={block_seconds} gives {block_samples} samples, "
            f"which is not divisible by model seq_len_multiple_of={frame_multiple}."
        )
    if context_samples % frame_multiple != 0:
        raise ValueError(
            f"--context-ms={context_ms} gives {context_samples} samples, "
            f"which is not divisible by model seq_len_multiple_of={frame_multiple}."
        )

    reconstructed_blocks = []
    for start in range(0, wave.shape[-1], block_samples):
        current = wave[..., start:(start + block_samples)]
        valid_samples = current.shape[-1]
        padded_length = (
            (valid_samples + frame_multiple - 1)
            // frame_multiple
            * frame_multiple
        )
        current_padded = F.pad(
            current,
            (0, padded_length - valid_samples)
        )

        previous_start = max(0, start - context_samples)
        context = wave[..., previous_start:start]
        codec_input = torch.cat((context, current_padded), dim=-1)
        if bypass_rvq:
            reconstructed = model.forward_bypass_rvq(
                codec_input.unsqueeze(0),
                return_recons_only=True,
            )
        else:
            reconstructed = model(
                codec_input.unsqueeze(0),
                return_recons_only=True,
                num_quantizers=num_quantizers,
            )
        reconstructed = reconstructed[
            ...,
            context.shape[-1]:(context.shape[-1] + valid_samples)
        ]
        reconstructed_blocks.append(reconstructed)

    return torch.cat(reconstructed_blocks, dim=-1)


def main() -> None:
    args = parse_args()

    if args.no_ema and args.weights != "auto":
        raise ValueError("Use either --no-ema or --weights, not both.")

    requested_weights = "online" if args.no_ema else args.weights
    checkpoint = args.checkpoint or latest_checkpoint(
        use_ema=(requested_weights == "ema")
    )
    if checkpoint is None:
        raise FileNotFoundError(
            "No checkpoint found. Pass --checkpoint E:\\lyra\\results\\...\\soundstream.<step>.pt"
        )

    pkg = load_checkpoint(checkpoint)
    checkpoint_weight_source = str(pkg.get("weight_source", "")).lower()
    if requested_weights == "auto":
        if checkpoint_weight_source.startswith("ema"):
            requested_weights = "ema"
        else:
            requested_weights = "online"
            if "ema_model" in pkg and not checkpoint_weight_source:
                print(
                    "WARNING: checkpoint contains both online and EMA weights; "
                    "--weights auto selected online. Pass --weights ema to "
                    "evaluate EMA explicitly."
                )
    use_ema = requested_weights == "ema"
    model_only_ema = pkg.get("weight_source") == "ema"
    if not use_ema and (
        checkpoint.name == "best_ema.pt" or model_only_ema
    ):
        online_checkpoint = checkpoint.with_name("best.pt")
        if not online_checkpoint.exists():
            raise FileNotFoundError(
                "Online weights were requested for an EMA-only checkpoint, but "
                f"the matching online checkpoint is missing: {online_checkpoint}"
            )
        checkpoint = online_checkpoint
        pkg = load_checkpoint(checkpoint)

    model = soundstream_from_checkpoint(pkg, use_ema=use_ema)
    model = model.to(args.device)
    model.eval()

    num_quantizers = (
        BITRATE_TO_QUANTIZERS[args.bitrate]
        if args.bitrate is not None
        else model.num_quantizers
    )
    if args.bitrate is not None and model.codebook_size != 256:
        raise ValueError(
            f"Lyra-style bitrate selection requires codebook_size=256, "
            f"but the checkpoint uses {model.codebook_size}."
        )
    if num_quantizers > model.num_quantizers:
        raise ValueError(
            f"{args.bitrate} bps requires {num_quantizers} RVQ quantizers, "
            f"but the checkpoint only has {model.num_quantizers}."
        )

    sample_rate = int(model.target_sample_hz)
    wave = load_audio(args.input_audio, target_sample_rate=sample_rate)
    original_num_samples = wave.shape[-1]
    frame_multiple = int(model.seq_len_multiple_of)

    with torch.inference_mode():
        wave = wave.to(args.device)
        if isinstance(model, FrameStreamingSoundStream):
            padded_num_samples = (
                (original_num_samples + frame_multiple - 1)
                // frame_multiple
                * frame_multiple
            )
            wave = F.pad(wave, (0, padded_num_samples - original_num_samples))
            recon = run_frame_streaming_codec(
                model,
                wave,
                num_quantizers,
                bypass_rvq=args.bypass_rvq,
            )
        elif args.whole_file:
            padded_num_samples = (
                (original_num_samples + frame_multiple - 1)
                // frame_multiple
                * frame_multiple
            )
            wave = F.pad(wave, (0, padded_num_samples - original_num_samples))
            if args.bypass_rvq:
                recon = model.forward_bypass_rvq(
                    wave.unsqueeze(0),
                    return_recons_only=True,
                )
            else:
                recon = model(
                    wave.unsqueeze(0),
                    return_recons_only=True,
                    num_quantizers=num_quantizers,
                )
        else:
            recon = run_block_context_codec(
                model,
                wave,
                num_quantizers=num_quantizers,
                block_seconds=args.block_seconds,
                context_ms=args.context_ms,
                bypass_rvq=args.bypass_rvq,
            )

    recon = recon.squeeze(0).detach().cpu()
    while recon.ndim > 2 and recon.shape[0] == 1:
        recon = recon.squeeze(0)
    if recon.ndim == 1:
        recon = recon.unsqueeze(0)
    if recon.ndim != 2:
        raise RuntimeError(f"Expected reconstructed audio to be 2D [channels, time], got shape {tuple(recon.shape)}")

    recon = recon[..., :original_num_samples]
    recon = recon.clamp(-1.0, 1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_audio(args.output, recon, sample_rate)

    is_model_only_ema = checkpoint.name == "best_ema.pt"
    weight_name = pkg.get(
        "weight_source",
        "EMA" if is_model_only_ema or (use_ema and "ema_model" in pkg) else "online"
    )
    print(f"Checkpoint: {checkpoint}")
    print(f"Weights: {weight_name}")
    print(f"Requested weight source: {requested_weights}")
    print(
        "Decoder upsample mode: "
        f"{getattr(model, 'decoder_upsample_mode', 'unknown')}"
    )
    print(
        "Decoder linear upsample kernel min: "
        f"{getattr(model, 'decoder_linear_upsample_kernel_min', 'unknown')}"
    )
    print(
        "Decoder residual scale: "
        f"{getattr(model, 'decoder_residual_scale', 'unknown')}"
    )
    print(f"Input: {args.input_audio}")
    print(f"Output: {args.output}")
    print(f"Sample rate: {sample_rate} Hz")
    bits_per_index = math.log2(model.codebook_size)
    bitrate = (
        sample_rate
        / model.seq_len_multiple_of
        * num_quantizers
        * bits_per_index
    )
    print(f"RVQ quantizers: {num_quantizers}/{model.num_quantizers}")
    print(f"Codec payload bitrate: {bitrate:.0f} bps")
    print(f"RVQ bypass diagnostic: {args.bypass_rvq}")
    if isinstance(model, FrameStreamingSoundStream):
        print(f"Inference mode: frame streaming ({model.stream_frame_size} samples/frame)")
    elif args.whole_file:
        print("Inference mode: whole file")
    else:
        print(
            "Inference mode: block/context "
            f"({args.block_seconds:.3f} s blocks, {args.context_ms:.3f} ms context)"
        )


if __name__ == "__main__":
    main()
