#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,7}"
NUM_PROCESSES="${NUM_PROCESSES:-7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29511}"
AUDIO_DIR="${AUDIO_DIR:-$PWD/data/librispeech/LibriSpeech/train-clean-100}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/results/stage25-rvq-midband-${RUN_TAG}-3k-7gpu-4s}"
LOG_FILE="${LOG_FILE:-$PWD/logs/stage25-rvq-midband-${RUN_TAG}-3k-7gpu-4s.log}"

if [[ -z "$INIT_CHECKPOINT" || ! -f "$INIT_CHECKPOINT" ]]; then
  echo "ERROR: set INIT_CHECKPOINT to an existing quality-gated Stage-2 checkpoint." >&2
  exit 1
fi
if [[ ! -d "$AUDIO_DIR" ]]; then
  echo "ERROR: audio directory does not exist: $AUDIO_DIR" >&2
  exit 1
fi
if [[ -e "$RESULTS_DIR" ]]; then
  echo "ERROR: results directory already exists: $RESULTS_DIR" >&2
  exit 1
fi

mkdir -p "$PWD/logs"
printf '%s\n' "$LOG_FILE" > "$PWD/logs/current_stage25_rvq_midband_log_file.txt"
printf '%s\n' "$RESULTS_DIR" > "$PWD/logs/current_stage25_rvq_midband_results_dir.txt"

mapfile -t DECODER_CONFIG < <(
  "$PYTHON_BIN" tools/checkpoint_decoder_config.py "$INIT_CHECKPOINT"
)
if [[ "${#DECODER_CONFIG[@]}" -ne 4 ]]; then
  echo "ERROR: could not read Decoder configuration from checkpoint." >&2
  exit 1
fi
if [[ "${DECODER_CONFIG[0]}" != "linear" || "${DECODER_CONFIG[3]}" != "0" ]]; then
  echo "ERROR: this controlled refinement requires the production baseline" >&2
  echo "       Decoder configuration mode=linear, split_first_upsample=0." >&2
  echo "       checkpoint reports mode=${DECODER_CONFIG[0]}," >&2
  echo "       split_first_upsample=${DECODER_CONFIG[3]}." >&2
  exit 1
fi
DECODER_SPLIT_FLAG="--no-decoder-split-first-upsample"

echo "===== Stage 2.5: Encoder + RVQ-only midband refinement ====="
echo "INIT_CHECKPOINT=$INIT_CHECKPOINT"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "steps 0-2999: Encoder LR=2e-8 + RVQ EMA; full Decoder frozen"
echo "generator optimizer warmup=100 steps; GAN disabled"
echo "active spectral weight=0.02 -> 0.03 over 300 steps"
echo "active spectral bands=1/1.5/1/0.5/0.25; frame phase=0.005"
echo "best_by_midband.pt is quality-gated; eval=100, save=200"

CUDA_VISIBLE_DEVICES="$GPU_LIST" accelerate launch \
  --multi_gpu \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --dynamo_backend no \
  --mixed_precision no \
  train_soundstream.py \
  --stage gan_pretrain \
  --stage25-rvq-midband-refine \
  --audio-dir "$AUDIO_DIR" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --results-dir "$RESULTS_DIR" \
  --decoder-upsample-mode "${DECODER_CONFIG[0]}" \
  --decoder-linear-upsample-kernel-min "${DECODER_CONFIG[1]}" \
  --decoder-interpolation-mode "${DECODER_CONFIG[2]}" \
  "$DECODER_SPLIT_FLAG" \
  --num-train-steps 3000 \
  --batch-size 4 \
  --grad-accum-every 1 \
  --segment-seconds 4.0 \
  --dl-num-workers 6 \
  --seed 42 \
  --save-model-every 200 \
  --best-eval-every 100 \
  --early-stopping-min-steps 1000 \
  --early-stopping-patience 10 \
  --si-sdr-loss-weight 0.05 \
  --si-sdr-loss-start-steps 0 \
  --si-sdr-loss-warmup-steps 0 \
  --spectral-envelope-loss-weight 0.05 \
  --spectral-envelope-loss-start-steps 0 \
  --spectral-envelope-loss-warmup-steps 0 \
  --voiced-highband-loss-weight 0.06 \
  --voiced-highband-energy-deficit-weight 0.40 \
  --voiced-highband-energy-margin-db 0.05 \
  --voiced-highband-loss-start-steps 0 \
  --voiced-highband-loss-warmup-steps 0 \
  --voiced-hf-retention-loss-weight 0.02 \
  --voiced-hf-retention-margin-db 0.50 \
  --upper-highband-loss-weight 0.0025 \
  --upper-highband-loss-start-steps 0 \
  --upper-highband-loss-warmup-steps 0 \
  --active-spectral-detail-loss-weight 0.03 \
  --active-spectral-detail-loss-start-steps 0 \
  --active-spectral-detail-loss-warmup-steps 300 \
  --frame-phase-loss-weight 0.005 \
  --frame-phase-loss-warmup-steps 0 \
  --noise-floor-loss-weight 0.03 \
  --click-loss-weight 0 \
  --jump-loss-weight 0 \
  --preemph-loss-weight 0 \
  --no-stage2-plateau-lr \
  --stage2-quality-retention-patience 4 \
  --stage2-rvq-retention-patience 2 \
  --stage2-max-aligned-si-sdr-drop 0.10 \
  --stage2-max-voiced-hf-ratio-db-drop 0.30 \
  --stage2-max-voiced-hf-ratio-db-rise 1.50 \
  --stage2-voiced-hf-score-weight 3.0 \
  --stage2-max-click-score-rise 0.30 \
  --stage2-max-ac320-isolated-rise 0.0015 \
  --stage2-max-comb-median-excess-db-rise 0.10 \
  --clean-gate-max-click-score 6.0 \
  --clean-gate-max-click-excess 0.5 \
  --test-eval-batches 10 \
  --no-resume \
  > "$LOG_FILE" 2>&1

echo "Stage 2.5 RVQ-midband refinement complete: $RESULTS_DIR"
