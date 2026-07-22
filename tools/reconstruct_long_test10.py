from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
AUDIO_EXTENSIONS = ("flac", "wav", "mp3", "webm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the longest files from the same speaker-held-out test split "
            "used by train_soundstream.py, then reconstruct them."
        )
    )
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-files", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--bitrate", type=int, default=9200)
    parser.add_argument("--block-seconds", type=float, default=5.0)
    parser.add_argument("--context-ms", type=float, default=60.0)
    parser.add_argument(
        "--weights", choices=("auto", "online", "ema"), default="auto"
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Create originals and manifest without running inference.",
    )
    parser.add_argument(
        "--whole-file",
        action="store_true",
        help="Use one whole-file model call instead of block/context inference.",
    )
    return parser.parse_args()


def all_audio_files(audio_dir: Path) -> list[Path]:
    # Keep the same extension-major ordering as SoundDataset.
    return sorted(
        file
        for extension in AUDIO_EXTENSIONS
        for file in audio_dir.glob(f"**/*.{extension}")
    )


def held_out_test_files(
    files: list[Path],
    audio_dir: Path,
    *,
    seed: int,
    valid_frac: float,
    test_frac: float,
) -> tuple[list[Path], list[str]]:
    if valid_frac < 0 or test_frac <= 0 or valid_frac + test_frac >= 1:
        raise ValueError("Require valid_frac >= 0, test_frac > 0, and sum < 1.")

    speaker_to_files: dict[str, list[Path]] = {}
    resolved_root = audio_dir.resolve()
    for file in files:
        relative_parts = file.resolve().relative_to(resolved_root).parts
        if len(relative_parts) < 2:
            raise ValueError(f"Audio is not inside a speaker directory: {file}")
        speaker_to_files.setdefault(relative_parts[0], []).append(file)

    speakers = sorted(speaker_to_files)
    permutation = torch.randperm(
        len(speakers), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    speakers = [speakers[index] for index in permutation]

    train_frac = 1.0 - valid_frac - test_frac
    train_count = int(round(train_frac * len(speakers)))
    valid_count = int(round(valid_frac * len(speakers)))
    test_speakers = speakers[train_count + valid_count :]
    test_files = [
        file
        for speaker in sorted(test_speakers)
        for file in speaker_to_files[speaker]
    ]
    return test_files, test_speakers


def audio_metadata(path: Path) -> tuple[float, int, int]:
    info = sf.info(str(path))
    if info.samplerate <= 0 or info.frames <= 0:
        raise ValueError(f"Invalid audio metadata: {path}")
    return info.frames / info.samplerate, info.samplerate, info.frames


def main() -> None:
    args = parse_args()
    audio_dir = args.audio_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.num_files <= 0:
        raise ValueError("--num-files must be positive.")

    files = all_audio_files(audio_dir)
    if not files:
        raise RuntimeError(f"No supported audio files found under: {audio_dir}")
    test_files, test_speakers = held_out_test_files(
        files,
        audio_dir,
        seed=args.seed,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
    )

    ranked: list[tuple[float, int, int, Path]] = []
    for index, source in enumerate(test_files, start=1):
        try:
            duration, sample_rate, frames = audio_metadata(source)
        except Exception as error:
            print(f"skip unreadable audio: {source}: {error}", file=sys.stderr)
            continue
        ranked.append((duration, sample_rate, frames, source))
        if index % 500 == 0:
            print(f"scanned {index}/{len(test_files)} test files", flush=True)

    ranked.sort(key=lambda row: (-row[0], str(row[3])))
    selected = ranked[: args.num_files]
    if len(selected) < args.num_files:
        raise RuntimeError(
            f"Requested {args.num_files} files but only {len(selected)} are readable."
        )

    originals_dir = output_dir / "originals"
    reconstructions_dir = output_dir / "reconstructions"
    originals_dir.mkdir(parents=True, exist_ok=True)
    reconstructions_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.tsv"
    inference_log_path = output_dir / "inference.log"

    rows = []
    for rank, (duration, sample_rate, frames, source) in enumerate(selected, start=1):
        prefix = f"{rank:02d}"
        copied = originals_dir / f"{prefix}_{source.name}"
        reconstructed = reconstructions_dir / f"{prefix}_{source.stem}_recon.flac"
        shutil.copy2(source, copied)
        rows.append(
            {
                "rank": rank,
                "duration_sec": f"{duration:.6f}",
                "sample_rate": sample_rate,
                "num_frames": frames,
                "speaker": source.resolve().relative_to(audio_dir).parts[0],
                "dataset_source": str(source),
                "original_copy": str(copied),
                "reconstruction": str(reconstructed),
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"dataset_files={len(files)}, test_speakers={len(test_speakers)}, "
        f"test_files={len(test_files)}"
    )
    print(f"manifest={manifest_path}")
    if args.select_only:
        print(f"selection complete: {output_dir}")
        return

    infer_script = PROJECT_DIR / "infer_soundstream.py"
    with inference_log_path.open("w", encoding="utf-8") as log_file:
        for row in rows:
            print(
                f"[{row['rank']}/{len(rows)}] reconstructing "
                f"{Path(row['original_copy']).name} "
                f"({row['duration_sec']} s)",
                flush=True,
            )
            command = [
                sys.executable,
                str(infer_script),
                row["original_copy"],
                "--checkpoint",
                str(checkpoint),
                "--output",
                row["reconstruction"],
                "--weights",
                args.weights,
                "--bitrate",
                str(args.bitrate),
                "--block-seconds",
                str(args.block_seconds),
                "--context-ms",
                str(args.context_ms),
            ]
            if args.whole_file:
                command.append("--whole-file")
            subprocess.run(
                command,
                cwd=PROJECT_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
            log_file.flush()

    print(f"reconstruction complete: {output_dir}")
    print(f"inference_log={inference_log_path}")


if __name__ == "__main__":
    main()
