#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,7}"
NUM_PROCESSES="${NUM_PROCESSES:-7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29507}"
AUDIO_DIR="${AUDIO_DIR:-$PWD/data/librispeech/LibriSpeech/train-clean-100}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/results/gan-stage25-encoder-refine-${RUN_TAG}-10k-7gpu-4s}"
LOG_FILE="${LOG_FILE:-$PWD/logs/gan-stage25-encoder-refine-${RUN_TAG}-10k-7gpu-4s.log}"

if [[ -z "$INIT_CHECKPOINT" ]]; then
  echo "ERROR: 请设置 INIT_CHECKPOINT 为阶段2 checkpoint（推荐 best_full_gan_balanced.pt、best_gan_balanced.pt 或试听确认的 soundstream.N.pt）。" >&2
  exit 1
fi
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "ERROR: checkpoint 不存在: $INIT_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -d "$AUDIO_DIR" ]]; then
  echo "ERROR: 音频目录不存在: $AUDIO_DIR" >&2
  exit 1
fi
if [[ -e "$RESULTS_DIR" ]]; then
  echo "ERROR: 结果目录已存在，阶段2.5要求使用新目录: $RESULTS_DIR" >&2
  exit 1
fi

mkdir -p "$PWD/logs"
printf '%s\n' "$LOG_FILE" > "$PWD/logs/current_stage25_log_file.txt"
printf '%s\n' "$RESULTS_DIR" > "$PWD/logs/current_stage25_results_dir.txt"
mapfile -t DECODER_CONFIG < <("$PYTHON_BIN" tools/checkpoint_decoder_config.py "$INIT_CHECKPOINT")
if [[ "${#DECODER_CONFIG[@]}" -ne 2 ]]; then
  echo "ERROR: 无法读取 checkpoint 的 Decoder 配置。" >&2
  exit 1
fi

echo "===== Stage 2.5: short Encoder + Decoder refinement ====="
echo "INIT_CHECKPOINT=$INIT_CHECKPOINT"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "GPU_LIST=$GPU_LIST"
echo "Decoder architecture: mode=${DECODER_CONFIG[0]}, kernel_min=${DECODER_CONFIG[1]}"

CUDA_VISIBLE_DEVICES="$GPU_LIST" accelerate launch \
  --multi_gpu \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --dynamo_backend no \
  --mixed_precision no \
  train_soundstream.py \
  --stage gan_pretrain \
  --stage25-encoder-refine \
  --audio-dir "$AUDIO_DIR" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --results-dir "$RESULTS_DIR" \
  --decoder-upsample-mode "${DECODER_CONFIG[0]}" \
  --decoder-linear-upsample-kernel-min "${DECODER_CONFIG[1]}" \
  --num-train-steps 10000 \
  --stage25-decoder-lr 2e-7 \
  --stage25-encoder-lr 1e-7 \
  --batch-size 4 \
  --grad-accum-every 1 \
  --segment-seconds 4.0 \
  --dl-num-workers 6 \
  --seed 42 \
  --save-model-every 1000 \
  --best-eval-every 500 \
  --early-stopping-min-steps 5000 \
  --early-stopping-patience 12 \
  --si-sdr-loss-weight 0.05 \
  --spectral-envelope-loss-weight 0.05 \
  --voiced-highband-loss-weight 0.06 \
  --voiced-highband-energy-deficit-weight 0.40 \
  --voiced-highband-energy-margin-db 0.05 \
  --voiced-hf-retention-loss-weight 0.02 \
  --voiced-hf-retention-margin-db 0.50 \
  --noise-floor-loss-weight 0.03 \
  --click-loss-weight 0 \
  --jump-loss-weight 0 \
  --preemph-loss-weight 0 \
  --no-stage2-plateau-lr \
  --waveform-discr-lrs 5e-7 5e-7 2.5e-7 \
  --stft-discr-lr 2.5e-7 \
  --waveform-discr-update-every 2 4 4 \
  --waveform-discr-loss-weights 1.0 0.25 0.25 \
  --stft-discr-update-every 4 \
  --stft-discr-loss-weight 0.5 \
  --gan-grad-diagnostics-every 500 \
  --discr-max-grad-norm 1.0 \
  --stage2-quality-retention-patience 8 \
  --stage2-rvq-retention-patience 6 \
  --stage2-max-aligned-si-sdr-drop 0.10 \
  --stage2-max-voiced-hf-ratio-db-drop 0.30 \
  --stage2-max-voiced-hf-ratio-db-rise 1.50 \
  --stage2-voiced-hf-score-weight 3.0 \
  --stage2-max-click-score-rise 0.30 \
  --clean-gate-max-click-score 6.0 \
  --clean-gate-max-click-excess 0.5 \
  --stft-r1-every 32 \
  --stft-r1-gamma 5e-3 \
  --waveform-r1-every 0 \
  --waveform-r1-gamma 0 \
  --test-eval-batches 10 \
  --no-resume \
  > "$LOG_FILE" 2>&1

echo "Stage 2.5 complete: $RESULTS_DIR"
