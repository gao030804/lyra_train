#!/usr/bin/env bash
set -euo pipefail

# Legacy filename retained for compatibility.  The active pipeline is:
# Stage 1 (reconstruction) -> Stage 2 (GAN), with no Stage 1.5 refinement.
cd "${LYRA_REPO_DIR:-$HOME/gyh/lyra_md}"
mkdir -p logs results

# Preserve the active conda environment's runtime libraries when launched by
# setsid/nohup.  Start this script from an already activated `(lyra)` shell.
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONUNBUFFERED=1

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"
NUM_PROCESSES="${NUM_PROCESSES:-6}"
AUDIO_DIR="${AUDIO_DIR:-$PWD/data/librispeech/LibriSpeech/train-clean-100}"
SEED="${SEED:-42}"
STAGE2_STEPS="${STAGE2_STEPS:-50000}"
# Controlled Stage-2 GAN ablations. Keep the production baseline unchanged
# unless one explicit single-variable experiment is selected:
#   baseline: unchanged G LR=5e-7 and adversarial max=2e-4
#   adv0003:  only adversarial max becomes 3e-4
#   glr075:   only post-hold G LR becomes 7.5e-7
STAGE2_ABLATION="${STAGE2_ABLATION:-baseline}"
case "$STAGE2_ABLATION" in
  baseline)
    STAGE2_GENERATOR_LR=5e-7
    STAGE2_GAN_ADVERSARIAL_MAX=2e-4
    ;;
  adv0003)
    STAGE2_GENERATOR_LR=5e-7
    STAGE2_GAN_ADVERSARIAL_MAX=3e-4
    ;;
  glr075)
    STAGE2_GENERATOR_LR=7.5e-7
    STAGE2_GAN_ADVERSARIAL_MAX=2e-4
    ;;
  *)
    echo "ERROR: STAGE2_ABLATION must be baseline, adv0003, or glr075." >&2
    exit 2
    ;;
esac
STAGE2_GAN_FEATURE_MAX=1.5
# Decoder experiment controls:
#   production baseline: DECODER_INTERPOLATION_MODE=linear, DECODER_SPLIT_FIRST_UPSAMPLE=0
#   split ablation:      DECODER_INTERPOLATION_MODE=linear, DECODER_SPLIT_FIRST_UPSAMPLE=1
#   cubic ablation:      DECODER_INTERPOLATION_MODE=cubic,  DECODER_SPLIT_FIRST_UPSAMPLE=0
DECODER_INTERPOLATION_MODE="${DECODER_INTERPOLATION_MODE:-linear}"
DECODER_SPLIT_FIRST_UPSAMPLE="${DECODER_SPLIT_FIRST_UPSAMPLE:-0}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"
RESUME_STAGE1="${RESUME_STAGE1:-0}"
FALLBACK_STAGE1_CKPT="${FALLBACK_STAGE1_CKPT:-}"
FALLBACK_MAX_SISDR_DROP="${FALLBACK_MAX_SISDR_DROP:-0.15}"
FALLBACK_MAX_CORR_DROP="${FALLBACK_MAX_CORR_DROP:-0.01}"
FALLBACK_MAX_CLICK_RISE="${FALLBACK_MAX_CLICK_RISE:-1.00}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"

# Short diagnostics intentionally finish before the formal 60k/90k schedule
# floors. Clamp those floors to the requested run length so the same launcher
# works for both a 30k diagnostic and a 200k formal run.
if (( STAGE2_STEPS < 20000 )); then
  DEFAULT_STAGE2_EARLY_STOP_MIN_STEPS="$STAGE2_STEPS"
else
  DEFAULT_STAGE2_EARLY_STOP_MIN_STEPS=20000
fi
if (( STAGE2_STEPS < 10000 )); then
  DEFAULT_STAGE2_PLATEAU_START_STEPS="$STAGE2_STEPS"
else
  DEFAULT_STAGE2_PLATEAU_START_STEPS=10000
fi
STAGE2_EARLY_STOP_MIN_STEPS="${STAGE2_EARLY_STOP_MIN_STEPS:-$DEFAULT_STAGE2_EARLY_STOP_MIN_STEPS}"
STAGE2_PLATEAU_START_STEPS="${STAGE2_PLATEAU_START_STEPS:-$DEFAULT_STAGE2_PLATEAU_START_STEPS}"

if [[ "$SKIP_STAGE1" != "0" && "$SKIP_STAGE1" != "1" ]]; then
    echo "ERROR: SKIP_STAGE1 must be 0 or 1." >&2
    exit 2
fi

if [[ "$DECODER_INTERPOLATION_MODE" != "linear" && "$DECODER_INTERPOLATION_MODE" != "cubic" ]]; then
  echo "ERROR: DECODER_INTERPOLATION_MODE must be linear or cubic." >&2
  exit 2
fi

if [[ "$DECODER_SPLIT_FIRST_UPSAMPLE" != "0" && "$DECODER_SPLIT_FIRST_UPSAMPLE" != "1" ]]; then
  echo "ERROR: DECODER_SPLIT_FIRST_UPSAMPLE must be 0 or 1." >&2
  exit 2
fi

if [[ "$DECODER_SPLIT_FIRST_UPSAMPLE" == "1" ]]; then
  DECODER_SPLIT_FLAG="--decoder-split-first-upsample"
else
  DECODER_SPLIT_FLAG="--no-decoder-split-first-upsample"
fi

if [[ "$RESUME_STAGE1" != "0" && "$RESUME_STAGE1" != "1" ]]; then
  echo "ERROR: RESUME_STAGE1 must be 0 or 1." >&2
  exit 2
fi

if [[ "$SKIP_STAGE1" == "1" && "$RESUME_STAGE1" == "1" ]]; then
  echo "ERROR: SKIP_STAGE1=1 and RESUME_STAGE1=1 cannot be used together." >&2
  exit 2
fi

if (( STAGE2_EARLY_STOP_MIN_STEPS < 0 || STAGE2_EARLY_STOP_MIN_STEPS > STAGE2_STEPS )); then
  echo "ERROR: STAGE2_EARLY_STOP_MIN_STEPS must be in [0, STAGE2_STEPS]." >&2
  exit 2
fi
if (( STAGE2_PLATEAU_START_STEPS < 0 || STAGE2_PLATEAU_START_STEPS > STAGE2_STEPS )); then
  echo "ERROR: STAGE2_PLATEAU_START_STEPS must be in [0, STAGE2_STEPS]." >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
if (( ${#GPU_IDS[@]} != NUM_PROCESSES )); then
  echo "ERROR: GPU_LIST has ${#GPU_IDS[@]} devices but NUM_PROCESSES=$NUM_PROCESSES." >&2
  exit 2
fi

# Defaults include a timestamp, so --no-resume cannot accidentally reuse an
# older checkpoint's best-score state.  Set explicit names only when needed.
STAGE1_NAME="${STAGE1_NAME:-recon-pretrain-dscnn-relu-fp-64d-8q-${RUN_TAG}-${NUM_PROCESSES}gpu-4s}"
STAGE2_NAME="${STAGE2_NAME:-gan-pretrain-dscnn-relu-fp-64d-8q-from-s1-${RUN_TAG}-s2-${STAGE2_ABLATION}-50k-${NUM_PROCESSES}gpu-4s}"

STAGE1_DIR="$PWD/results/$STAGE1_NAME"
STAGE2_DIR="$PWD/results/$STAGE2_NAME"
STAGE1_LOG="$PWD/logs/$STAGE1_NAME.log"
STAGE2_LOG="$PWD/logs/$STAGE2_NAME.log"

require_fresh_dir() {
  local path="$1"
  local label="$2"

  if [ -e "$path" ]; then
    echo "ERROR: $label results directory already exists: $path" >&2
    echo "Use a new RUN_TAG / result name, or intentionally remove that directory first." >&2
    exit 2
  fi
}

pick_validation_best() {
  local results_dir="$1"
  local stage_label="$2"
  local checkpoint

  for checkpoint in \
    best_by_clarity.pt \
    best_selected.pt \
    best_by_aligned_si_sdr.pt; do
    if [ -f "$results_dir/$checkpoint" ]; then
      printf '%s\n' "$results_dir/$checkpoint"
      return 0
    fi
  done

  echo "ERROR: $stage_label finished without a clean-gated validation checkpoint in $results_dir" >&2
  echo "Raw diagnostic checkpoints are intentionally not eligible for Stage 2 initialization." >&2
  return 1
}

checkpoint_decoder_args() {
  local checkpoint="$1"
  local -n output_array="$2"

  mapfile -t output_array < <(
    "$PYTHON_BIN" tools/checkpoint_decoder_config.py "$checkpoint"
  )
  if (( ${#output_array[@]} != 4 )); then
    echo "ERROR: could not read decoder architecture from $checkpoint" >&2
    return 1
  fi
}

evaluate_stage1_checkpoint() {
  local checkpoint="$1"
  local report_file="$2"
  local -a decoder_args
  local decoder_split_flag

  checkpoint_decoder_args "$checkpoint" decoder_args
  if [[ "${decoder_args[3]}" == "1" ]]; then
    decoder_split_flag="--decoder-split-first-upsample"
  else
    decoder_split_flag="--no-decoder-split-first-upsample"
  fi
  echo "Fixed-validation evaluation: $checkpoint"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$PYTHON_BIN" train_soundstream.py \
    --stage recon_pretrain \
    --audio-dir "$AUDIO_DIR" \
    --results-dir "$STAGE1_DIR" \
    --batch-size 4 \
    --segment-seconds 4.0 \
    --dl-num-workers 6 \
    --seed "$SEED" \
    --decoder-upsample-mode "${decoder_args[0]}" \
    --decoder-linear-upsample-kernel-min "${decoder_args[1]}" \
    --decoder-interpolation-mode "${decoder_args[2]}" \
    "$decoder_split_flag" \
    --validation-only \
    --validation-checkpoint "$checkpoint" \
    --validation-report-file "$report_file" \
    --no-resume
}

run_stage() {
  local stage_label="$1"
  local port="$2"
  shift 2

  echo "===== $stage_label ====="
  echo "GPU_LIST=$GPU_LIST"
  echo "NUM_PROCESSES=$NUM_PROCESSES"
  echo "AUDIO_DIR=$AUDIO_DIR"

  CUDA_VISIBLE_DEVICES="$GPU_LIST" accelerate launch \
    --multi_gpu \
    --num_processes "$NUM_PROCESSES" \
    --num_machines 1 \
    --main_process_port "$port" \
    --dynamo_backend no \
    --mixed_precision no \
    train_soundstream.py "$@"
}

run_and_log() {
  local log_file="$1"
  local append_log="$2"
  shift 2

  if [[ "$append_log" == "1" ]]; then
    "$@" >> "$log_file" 2>&1
  else
    "$@" > "$log_file" 2>&1
  fi
}

# ----- Stage 1: reconstruction pretraining -----
if [[ "$SKIP_STAGE1" == "1" ]]; then
  echo "===== Stage 1: skipped; using an existing validation-selected checkpoint ====="
  echo "STAGE1_DIR=$STAGE1_DIR"
  STAGE1_CKPT="$(pick_validation_best "$STAGE1_DIR" "Skipped Stage 1")"
  echo "Stage 1 best checkpoint: $STAGE1_CKPT"
else
if [[ "$RESUME_STAGE1" == "1" ]]; then
  if [[ ! -f "$STAGE1_DIR/latest.pt" ]]; then
    echo "ERROR: RESUME_STAGE1=1 requires $STAGE1_DIR/latest.pt" >&2
    exit 2
  fi
  echo "===== Stage 1: resuming from latest.pt ====="
  STAGE1_RESUME_FLAG="--resume"
else
  require_fresh_dir "$STAGE1_DIR" "Stage 1"
  STAGE1_RESUME_FLAG="--no-resume"
fi
echo "STAGE1_DIR=$STAGE1_DIR"
echo "STAGE1_LOG=$STAGE1_LOG"
echo "Stage 1 decoder architecture: interpolation=$DECODER_INTERPOLATION_MODE, split_first_x8=$DECODER_SPLIT_FIRST_UPSAMPLE"
run_and_log "$STAGE1_LOG" "$RESUME_STAGE1" run_stage "Stage 1: recon_pretrain" 29501 \
  --stage recon_pretrain \
  --audio-dir "$AUDIO_DIR" \
  --results-dir "$STAGE1_DIR" \
  --num-train-steps 150000 \
  --batch-size 4 \
  --grad-accum-every 1 \
  --segment-seconds 4.0 \
  --dl-num-workers 6 \
  --seed "$SEED" \
  --decoder-interpolation-mode "$DECODER_INTERPOLATION_MODE" \
  "$DECODER_SPLIT_FLAG" \
  --decoder-residual-scale-start 0.2 \
  --decoder-residual-scale-end 1.0 \
  --decoder-residual-scale-warmup-start-steps 0 \
  --decoder-residual-scale-warmup-end-steps 15000 \
  --si-sdr-loss-weight 0.07 \
  --si-sdr-loss-start-steps 5000 \
  --si-sdr-loss-warmup-steps 15000 \
  --spectral-envelope-loss-weight 0.08 \
  --spectral-envelope-loss-start-steps 5000 \
  --spectral-envelope-loss-warmup-steps 15000 \
  --formant-peak-loss-weight 0.02 \
  --formant-peak-loss-start-steps 15000 \
  --formant-peak-loss-warmup-steps 20000 \
  --stft-recon-loss-weight 0.05 \
  --stft-recon-loss-start-steps 5000 \
  --stft-recon-loss-warmup-steps 15000 \
  --loss-grad-diagnostics-every 1000 \
  --voiced-highband-loss-weight 0.04 \
  --voiced-highband-loss-start-steps 5000 \
  --voiced-highband-loss-warmup-steps 15000 \
  --voiced-highband-energy-deficit-weight 0.35 \
  --voiced-highband-energy-margin-db 0.10 \
  --stage1-plateau-lr \
  --plateau-start-steps 60000 \
  --plateau-factor 0.5 \
  --plateau-patience 16 \
  --plateau-threshold 0.03 \
  --plateau-cooldown 2 \
  --plateau-min-lr 1e-5 \
  --plateau-unclean-grace-checks 8 \
  --click-loss-weight 0.002 \
  --jump-loss-weight 0 \
  --preemph-loss-weight 0 \
  --noise-floor-loss-weight 0.03 \
  --transient-loss-warmup-steps 5000 \
  --test-eval-batches 100 \
  "$STAGE1_RESUME_FLAG"

STAGE1_CKPT="$(pick_validation_best "$STAGE1_DIR" "Stage 1")"
echo "Stage 1 best checkpoint: $STAGE1_CKPT"
fi

# Optional automatic fallback.  Both checkpoints are scored on the same fixed
# validation waves; the held-out test split is never touched.  If the new run
# loses too much reconstruction / clarity quality or has unhealthy q00/q01/RVQ,
# Stage 2 is initialized from the established fallback instead.
if [[ -n "$FALLBACK_STAGE1_CKPT" ]]; then
  FALLBACK_STAGE1_CKPT="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$FALLBACK_STAGE1_CKPT")"
  if [[ ! -f "$FALLBACK_STAGE1_CKPT" ]]; then
    echo "ERROR: FALLBACK_STAGE1_CKPT not found: $FALLBACK_STAGE1_CKPT" >&2
    exit 2
  fi

  if [[ "$STAGE1_CKPT" == "$FALLBACK_STAGE1_CKPT" ]]; then
    echo "Automatic fallback comparison skipped: new and fallback paths are identical."
  else
    NEW_VALIDATION_REPORT="$STAGE1_DIR/auto_fallback_new_validation.tsv"
    FALLBACK_VALIDATION_REPORT="$STAGE1_DIR/auto_fallback_reference_validation.tsv"
    FALLBACK_DECISION_REPORT="$STAGE1_DIR/auto_fallback_decision.txt"

    echo "===== Stage 1 automatic fallback evaluation ====="
    evaluate_stage1_checkpoint "$STAGE1_CKPT" "$NEW_VALIDATION_REPORT"
    evaluate_stage1_checkpoint "$FALLBACK_STAGE1_CKPT" "$FALLBACK_VALIDATION_REPORT"

    STAGE1_CKPT="$("$PYTHON_BIN" tools/select_stage1_checkpoint.py \
      --new-checkpoint "$STAGE1_CKPT" \
      --fallback-checkpoint "$FALLBACK_STAGE1_CKPT" \
      --new-report "$NEW_VALIDATION_REPORT" \
      --fallback-report "$FALLBACK_VALIDATION_REPORT" \
      --decision-report "$FALLBACK_DECISION_REPORT" \
      --max-si-sdr-drop "$FALLBACK_MAX_SISDR_DROP" \
      --max-correlation-drop "$FALLBACK_MAX_CORR_DROP" \
      --max-click-rise "$FALLBACK_MAX_CLICK_RISE")"
    echo "Automatic fallback decision report: $FALLBACK_DECISION_REPORT"
    cat "$FALLBACK_DECISION_REPORT"
  fi
else
  echo "Automatic fallback disabled: FALLBACK_STAGE1_CKPT is not set."
fi

# Stage 2 is not a codebook-repair stage. Re-evaluate the selected handoff on
# the same fixed validation set and reject it before allocating a fresh Stage-2
# directory if q00/RVQ or clipping are unhealthy.
STAGE2_PREFLIGHT_REPORT="$STAGE1_DIR/stage2_init_preflight_validation.tsv"
echo "===== Stage 2 initialization preflight ====="
evaluate_stage1_checkpoint "$STAGE1_CKPT" "$STAGE2_PREFLIGHT_REPORT"
"$PYTHON_BIN" tools/validate_stage2_init.py "$STAGE2_PREFLIGHT_REPORT" \
  --min-aligned-si-sdr 0 \
  --min-aligned-correlation 0.65 \
  --max-click-excess 0.5 \
  --min-q00-active-ratio 0.70 \
  --min-q00-perplexity 50 \
  --max-recon-clip-fraction 0.001

declare -a STAGE2_DECODER_ARGS
checkpoint_decoder_args "$STAGE1_CKPT" STAGE2_DECODER_ARGS
if [[ "${STAGE2_DECODER_ARGS[3]}" == "1" ]]; then
  STAGE2_DECODER_SPLIT_FLAG="--decoder-split-first-upsample"
else
  STAGE2_DECODER_SPLIT_FLAG="--no-decoder-split-first-upsample"
fi
echo "Stage 2 initialization checkpoint: $STAGE1_CKPT"
echo "Stage 2 decoder architecture: mode=${STAGE2_DECODER_ARGS[0]}, kernel_min=${STAGE2_DECODER_ARGS[1]}, interpolation=${STAGE2_DECODER_ARGS[2]}, split_first_x8=${STAGE2_DECODER_ARGS[3]}"

# ----- Stage 2: direct GAN refinement from the Stage-1 validation best -----
require_fresh_dir "$STAGE2_DIR" "Stage 2"
echo "STAGE2_DIR=$STAGE2_DIR"
echo "STAGE2_LOG=$STAGE2_LOG"
echo "STAGE2_STEPS=$STAGE2_STEPS"
echo "STAGE2_ABLATION=$STAGE2_ABLATION"
echo "STAGE2_GENERATOR_LR=$STAGE2_GENERATOR_LR"
echo "STAGE2_GAN_ADVERSARIAL_MAX=$STAGE2_GAN_ADVERSARIAL_MAX"
echo "STAGE2_GAN_FEATURE_MAX=$STAGE2_GAN_FEATURE_MAX"
echo "STAGE2_EARLY_STOP_MIN_STEPS=$STAGE2_EARLY_STOP_MIN_STEPS"
echo "STAGE2_PLATEAU_START_STEPS=$STAGE2_PLATEAU_START_STEPS"
run_stage "Stage 2: gan_pretrain" 29503 \
  --stage gan_pretrain \
  --audio-dir "$AUDIO_DIR" \
  --results-dir "$STAGE2_DIR" \
  --init-checkpoint "$STAGE1_CKPT" \
  --decoder-upsample-mode "${STAGE2_DECODER_ARGS[0]}" \
  --decoder-linear-upsample-kernel-min "${STAGE2_DECODER_ARGS[1]}" \
  --decoder-interpolation-mode "${STAGE2_DECODER_ARGS[2]}" \
  "$STAGE2_DECODER_SPLIT_FLAG" \
  --num-train-steps "$STAGE2_STEPS" \
  --generator-lr "$STAGE2_GENERATOR_LR" \
  --gan-adversarial-max "$STAGE2_GAN_ADVERSARIAL_MAX" \
  --gan-feature-max "$STAGE2_GAN_FEATURE_MAX" \
  --batch-size 4 \
  --grad-accum-every 1 \
  --segment-seconds 4.0 \
  --dl-num-workers 6 \
  --seed "$SEED" \
  --save-model-every 5000 \
  --best-eval-every 500 \
  --early-stopping-min-steps "$STAGE2_EARLY_STOP_MIN_STEPS" \
  --early-stopping-patience 12 \
  --si-sdr-loss-weight 0.05 \
  --si-sdr-loss-start-steps 0 \
  --si-sdr-loss-warmup-steps 0 \
  --spectral-envelope-loss-weight 0.05 \
  --formant-peak-loss-weight 0.01 \
  --formant-peak-loss-start-steps 0 \
  --formant-peak-loss-warmup-steps 5000 \
  --loss-grad-diagnostics-every 1000 \
  --voiced-highband-loss-weight 0.06 \
  --voiced-highband-energy-deficit-weight 0.40 \
  --voiced-highband-energy-margin-db 0.05 \
  --voiced-hf-retention-loss-weight 0.02 \
  --voiced-hf-retention-margin-db 0.50 \
  --voiced-highband-loss-start-steps 0 \
  --voiced-highband-loss-warmup-steps 0 \
  --upper-highband-loss-weight 0.0025 \
  --upper-highband-energy-deficit-weight 0.20 \
  --upper-highband-energy-margin-db 0.50 \
  --upper-highband-loss-start-steps 1000 \
  --upper-highband-loss-warmup-steps 5000 \
  --active-spectral-detail-loss-weight 0.02 \
  --active-spectral-detail-loss-start-steps 2000 \
  --active-spectral-detail-loss-warmup-steps 8000 \
  --stage2-active-spectral-score-weight 0.05 \
  --noise-floor-loss-weight 0.03 \
  --click-loss-weight 0 \
  --jump-loss-weight 0 \
  --preemph-loss-weight 0 \
  --stage2-plateau-lr \
  --stage2-plateau-start-steps "$STAGE2_PLATEAU_START_STEPS" \
  --stage2-plateau-factor 0.5 \
  --stage2-plateau-patience 8 \
  --stage2-plateau-threshold 0.01 \
  --stage2-plateau-cooldown 2 \
  --stage2-plateau-min-lr 2e-7 \
  --stage2-plateau-discr-min-lr 1e-7 \
  --stage2-plateau-stft-discr-min-lr 1e-7 \
  --waveform-discr-lrs 5e-7 5e-7 2.5e-7 \
  --stft-discr-lr 2.5e-7 \
  --waveform-discr-update-every 2 4 4 \
  --waveform-discr-loss-weights 1.0 0.25 0.25 \
  --stft-discr-update-every 4 \
  --stft-discr-loss-weight 0.5 \
  --gan-grad-diagnostics-every 500 \
  --discr-max-grad-norm 1.0 \
  --stage2-generator-freeze-steps 1000 \
  --stage2-discriminator-start-steps 0 \
  --stage2-unfreeze-encoder-rvq-step -1 \
  --stage2-generator-hold-steps 5000 \
  --stage2-generator-hold-lr 1e-7 \
  --stage2-discriminator-hold-steps 0 \
  --stage2-discriminator-hold-lr 5e-6 \
  --stage2-quality-gate-start-steps 20000 \
  --stage2-best-checkpoint-min-step 5000 \
  --stage2-quality-retention-patience 8 \
  --stage2-rvq-retention-patience 6 \
  --stage2-balanced-max-aligned-si-sdr-drop 0.10 \
  --stage2-max-click-score-rise 0.30 \
  --clean-gate-max-click-score 6.0 \
  --clean-gate-max-click-excess 0.5 \
  --stft-r1-every 32 \
  --stft-r1-gamma 5e-3 \
  --waveform-r1-every 0 \
  --waveform-r1-gamma 0 \
  --test-eval-batches 100 \
  --no-resume \
  > "$STAGE2_LOG" 2>&1

echo "===== Stage 1 -> 2 complete ====="
echo "Stage 1  results: $STAGE1_DIR"
echo "Stage 2  results: $STAGE2_DIR"
