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
            "best_ema.pt",
            "best.pt",
        )
        if use_ema
        else ("best.pt",)
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
        "--no-ema",
        action="store_true",
        help="Use the online model weights instead of EMA weights.",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        choices=tuple(BITRATE_TO_QUANTIZERS),
        default=None,
        help="Codec payload bitrate. Defaults to all RVQ quantizers in the checkpoint.",
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

    model.load_state_dict(state_dict, strict=True)
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
) -> torch.Tensor:
    encoder_state = None
    decoder_state = None
    reconstructed_frames = []
    batched_wave = wave.unsqueeze(0)

    for frame in batched_wave.split(model.stream_frame_size, dim=-1):
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


def main() -> None:
    args = parse_args()

    use_ema = not args.no_ema
    checkpoint = args.checkpoint or latest_checkpoint(use_ema=use_ema)
    if checkpoint is None:
        raise FileNotFoundError(
            "No checkpoint found. Pass --checkpoint E:\\lyra\\results\\...\\soundstream.<step>.pt"
        )

    pkg = load_checkpoint(checkpoint)
    model_only_ema = pkg.get("weight_source") == "ema"
    if args.no_ema and (
        checkpoint.name == "best_ema.pt" or model_only_ema
    ):
        online_checkpoint = checkpoint.with_name("best.pt")
        if not online_checkpoint.exists():
            raise FileNotFoundError(
                f"--no-ema requires the matching online checkpoint: {online_checkpoint}"
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
    padded_num_samples = (
        (original_num_samples + frame_multiple - 1)
        // frame_multiple
        * frame_multiple
    )
    wave = F.pad(wave, (0, padded_num_samples - original_num_samples))

    with torch.inference_mode():
        wave = wave.to(args.device)
        if isinstance(model, FrameStreamingSoundStream):
            recon = run_frame_streaming_codec(model, wave, num_quantizers)
        else:
            recon = model(
                wave.unsqueeze(0),
                return_recons_only=True,
                num_quantizers=num_quantizers,
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


if __name__ == "__main__":
    main()
