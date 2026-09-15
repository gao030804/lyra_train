#!/usr/bin/env bash
set -euo pipefail

# Deployment-oriented training:
#   A1 Encoder -> Decoder reconstruction (RVQ bypassed)
#   A1.5 low-LR Encoder/Decoder formant refinement (RVQ bypassed)
#   A2 Encoder -> Decoder GAN refinement (RVQ bypassed)
#   optional A2.5 persistent-state alignment before RVQ calibration
#   Q0 PCA initialization + projection-only reconstruction pretraining
#   B1 Encoder -> RVQ EMA calibration (Encoder/Decoder weights frozen)
#   B1.5 non-GAN adaptation (Decoder-only / RVQ EMA / Decoder polish)
#   B2 Encoder -> frozen RVQ -> Decoder GAN adaptation (Encoder/RVQ frozen)
#   final persistent-state Decoder fine-tune (Encoder/RVQ frozen, no GAN)

cd "${LYRA_REPO_DIR:-$HOME/gyh/lyra_md}"
mkdir -p logs results
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONUNBUFFERED=1

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"
NUM_PROCESSES="${NUM_PROCESSES:-6}"
AUDIO_DIR="${AUDIO_DIR:-$PWD/data/librispeech/LibriSpeech/train-clean-100}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
SEED="${SEED:-42}"
RVQ_LOOKUP_DIM="${RVQ_LOOKUP_DIM:-32}"
RVQ_DISTANCE="${RVQ_DISTANCE:-euclidean}"
RVQ_NUM_QUANTIZERS="${RVQ_NUM_QUANTIZERS:-9}"
RVQ_CODEBOOK_SIZES="${RVQ_CODEBOOK_SIZES:-256,128,128,128,128,128,128,128,128}"
BYPASS_STAGE1_STEPS="${BYPASS_STAGE1_STEPS:-150000}"
BYPASS_STAGE1_EARLY_STOPPING_MIN_STEPS="${BYPASS_STAGE1_EARLY_STOPPING_MIN_STEPS:-60000}"
BYPASS_FORMANT_REFINE_STEPS="${BYPASS_FORMANT_REFINE_STEPS:-20000}"
BYPASS_STAGE2_STEPS="${BYPASS_STAGE2_STEPS:-150000}"
RVQ_PROJECTION_STEPS="${RVQ_PROJECTION_STEPS:-20000}"
RVQ_PROJECTION_PCA_BATCHES="${RVQ_PROJECTION_PCA_BATCHES:-100}"
RVQ_PROJECTION_MIN_EVR="${RVQ_PROJECTION_MIN_EVR:-0.90}"
RVQ_PROJECTION_MIN_ALIGNED_SI_SDR="${RVQ_PROJECTION_MIN_ALIGNED_SI_SDR:-4.8}"
RVQ_PROJECTION_LR="${RVQ_PROJECTION_LR:-5e-5}"
RVQ_PROJECTION_LATENT_MSE_WEIGHT="${RVQ_PROJECTION_LATENT_MSE_WEIGHT:-0.50}"
RVQ_PROJECTION_LATENT_COSINE_WEIGHT="${RVQ_PROJECTION_LATENT_COSINE_WEIGHT:-0.15}"
RVQ_PROJECTION_ORTH_WEIGHT="${RVQ_PROJECTION_ORTH_WEIGHT:-0.05}"
RVQ_PROJECTION_TIE_WEIGHT="${RVQ_PROJECTION_TIE_WEIGHT:-0.05}"
RVQ_PROJECTION_FREEZE_INPUT_STEPS="${RVQ_PROJECTION_FREEZE_INPUT_STEPS:-3000}"
RVQ_PROJECTION_EARLY_STOPPING_MIN_STEPS="${RVQ_PROJECTION_EARLY_STOPPING_MIN_STEPS:-3000}"
RVQ_PROJECTION_EARLY_STOPPING_PATIENCE="${RVQ_PROJECTION_EARLY_STOPPING_PATIENCE:-10}"
RVQ_CALIBRATION_STEPS="${RVQ_CALIBRATION_STEPS:-1000}"
RVQ_CALIBRATION_KMEANS_BATCHES="${RVQ_CALIBRATION_KMEANS_BATCHES:-100}"
RVQ_B1_MIN_ALIGNED_SI_SDR="${RVQ_B1_MIN_ALIGNED_SI_SDR:-3.0}"
RVQ_B1_MIN_QUANTIZATION_GAP_DB="${RVQ_B1_MIN_QUANTIZATION_GAP_DB:--2.0}"
RVQ_B15_MAX_LOOKUP_NMSE="${RVQ_B15_MAX_LOOKUP_NMSE:-0.07}"
RVQ_B15_MAX_LATENT64_NMSE="${RVQ_B15_MAX_LATENT64_NMSE:-0.10}"
RVQ_B15_MIN_ALIGNED_SI_SDR="${RVQ_B15_MIN_ALIGNED_SI_SDR:-1.8}"
RVQ_JOINT_ADAPT_STEPS="${RVQ_JOINT_ADAPT_STEPS:-60000}"
RVQ_B15_DECODER_ONLY_STEPS="${RVQ_B15_DECODER_ONLY_STEPS:-10000}"
RVQ_B15_RVQ_ADAPT_END_STEPS="${RVQ_B15_RVQ_ADAPT_END_STEPS:-45000}"
RVQ_B15_POLISH_DECODER_LR="${RVQ_B15_POLISH_DECODER_LR:-1.5e-6}"
RVQ_STAGE2_STEPS="${RVQ_STAGE2_STEPS:-75000}"
ENABLE_PRE_RVQ_STATE_FT="${ENABLE_PRE_RVQ_STATE_FT:-0}"
PRE_RVQ_STATE_STEPS="${PRE_RVQ_STATE_STEPS:-20000}"
FINAL_STATE_STEPS="${FINAL_STATE_STEPS:-10000}"
RESUME_BYPASS_STAGE1="${RESUME_BYPASS_STAGE1:-0}"
START_PHASE="${START_PHASE:-bypass_stage1}"
STOP_AFTER_PHASE="${STOP_AFTER_PHASE:-}"
BYPASS_STAGE1_CKPT="${BYPASS_STAGE1_CKPT:-}"
BYPASS_STAGE2_CKPT="${BYPASS_STAGE2_CKPT:-}"
PRETRAINED_Q0_CKPT="${PRETRAINED_Q0_CKPT:-}"
PRETRAINED_RVQ_B15_FIDELITY_CKPT="${PRETRAINED_RVQ_B15_FIDELITY_CKPT:-}"
PRECEDING_CONTEXT_SECONDS="${PRECEDING_CONTEXT_SECONDS:-0.5}"

case "$RVQ_LOOKUP_DIM" in
  16|24|32) ;;
  *) echo "ERROR: RVQ_LOOKUP_DIM must be 16, 24, or 32." >&2; exit 2 ;;
esac
case "$RVQ_DISTANCE" in
  cosine|euclidean) ;;
  *) echo "ERROR: RVQ_DISTANCE must be cosine or euclidean." >&2; exit 2 ;;
esac

IFS=',' read -r -a RVQ_CODEBOOK_SIZE_VALUES <<< "$RVQ_CODEBOOK_SIZES"
if (( ${#RVQ_CODEBOOK_SIZE_VALUES[@]} != RVQ_NUM_QUANTIZERS )); then
  echo "ERROR: RVQ_CODEBOOK_SIZES must contain RVQ_NUM_QUANTIZERS comma-separated values." >&2
  echo "received: $RVQ_CODEBOOK_SIZES" >&2
  exit 2
fi
for SIZE in "${RVQ_CODEBOOK_SIZE_VALUES[@]}"; do
  if (( SIZE <= 1 || (SIZE & (SIZE - 1)) != 0 )); then
    echo "ERROR: every RVQ codebook size must be a power of two > 1; got $SIZE." >&2
    exit 2
  fi
done

if (( BYPASS_STAGE1_EARLY_STOPPING_MIN_STEPS > BYPASS_STAGE1_STEPS )); then
  BYPASS_STAGE1_EARLY_STOPPING_MIN_STEPS="$BYPASS_STAGE1_STEPS"
fi

RVQ_TOPOLOGY_TAG="${RVQ_CODEBOOK_SIZES//,/x}"
BASE="lowrank-dscnn-b234-relu-convtranspose-fp-64d-${RVQ_NUM_QUANTIZERS}q${RVQ_TOPOLOGY_TAG}-l${RVQ_LOOKUP_DIM}-${RVQ_DISTANCE}-${RUN_TAG}"
BYPASS_S1_DIR="$PWD/results/bypass-recon-$BASE"
BYPASS_FORMANT_DIR="$PWD/results/bypass-formant-refine-$BASE"
BYPASS_S2_DIR="$PWD/results/bypass-gan-$BASE"
RVQ_PROJECTION_DIR="$PWD/results/rvq-projection-$BASE"
RVQ_S1_DIR="$PWD/results/rvq-calibration-$BASE"
RVQ_B15_DIR="$PWD/results/rvq-joint-adapt-$BASE"
RVQ_B15_POLISH_DIR="$PWD/results/rvq-decoder-polish-$BASE"
RVQ_S2_DIR="$PWD/results/rvq-gan-$BASE"
PRE_RVQ_STATE_DIR="$PWD/results/state-pre-rvq-$BASE"
FINAL_STATE_DIR="$PWD/results/state-final-rvq-$BASE"

if [[ "$ENABLE_PRE_RVQ_STATE_FT" != "0" && "$ENABLE_PRE_RVQ_STATE_FT" != "1" ]]; then
  echo "ERROR: ENABLE_PRE_RVQ_STATE_FT must be 0 or 1." >&2
  exit 2
fi

# The deployable graph is Encoder -> RVQ -> Decoder with persistent 20 ms
# state.  Therefore the post-RVQ state pass is not optional: a zero/negative
# length would silently leave B2's offline checkpoint as the final artifact.
if (( FINAL_STATE_STEPS <= 0 )); then
  echo "ERROR: FINAL_STATE_STEPS must be positive; post-RVQ state alignment is mandatory." >&2
  exit 2
fi
if (( RVQ_JOINT_ADAPT_STEPS < RVQ_B15_RVQ_ADAPT_END_STEPS )); then
  echo "ERROR: RVQ_JOINT_ADAPT_STEPS must reach RVQ_B15_RVQ_ADAPT_END_STEPS." >&2
  exit 2
fi
RVQ_B15_POLISH_STEPS=$((RVQ_JOINT_ADAPT_STEPS - RVQ_B15_RVQ_ADAPT_END_STEPS))
if (( RVQ_B15_POLISH_STEPS <= 0 )); then
  echo "ERROR: B1.5 total steps must leave a positive Decoder-polish segment." >&2
  exit 2
fi
if (( RVQ_STAGE2_STEPS <= 0 )); then
  echo "ERROR: RVQ_STAGE2_STEPS must be positive." >&2
  exit 2
fi

if [[ "$START_PHASE" != "bypass_stage1" && \
      "$START_PHASE" != "bypass_formant_refine" && \
      "$START_PHASE" != "rvq_stage1" && \
      "$START_PHASE" != "rvq_polish" ]]; then
  echo "ERROR: START_PHASE must be bypass_stage1, bypass_formant_refine, rvq_stage1, or rvq_polish." >&2
  exit 2
fi

if [[ -n "$STOP_AFTER_PHASE" && \
      "$STOP_AFTER_PHASE" != "bypass_stage1" && \
      "$STOP_AFTER_PHASE" != "bypass_formant_refine" && \
      "$STOP_AFTER_PHASE" != "bypass_stage2" ]]; then
  echo "ERROR: STOP_AFTER_PHASE must be empty, bypass_stage1, bypass_formant_refine, or bypass_stage2." >&2
  exit 2
fi

if [[ "$RESUME_BYPASS_STAGE1" != "0" && "$RESUME_BYPASS_STAGE1" != "1" ]]; then
  echo "ERROR: RESUME_BYPASS_STAGE1 must be 0 or 1." >&2
  exit 2
fi

B1_Q0_REUSE_FLAGS=()
if [[ "$START_PHASE" == "rvq_polish" ]]; then
  if [[ "$RESUME_BYPASS_STAGE1" != "0" ]]; then
    echo "ERROR: RESUME_BYPASS_STAGE1 is not valid with START_PHASE=rvq_polish." >&2
    exit 2
  fi
  if [[ -z "$PRETRAINED_RVQ_B15_FIDELITY_CKPT" || ! -f "$PRETRAINED_RVQ_B15_FIDELITY_CKPT" ]]; then
    echo "ERROR: START_PHASE=rvq_polish requires PRETRAINED_RVQ_B15_FIDELITY_CKPT." >&2
    echo "received: ${PRETRAINED_RVQ_B15_FIDELITY_CKPT:-<empty>}" >&2
    exit 2
  fi
  RVQ_B15_FIDELITY_CKPT="$(readlink -f "$PRETRAINED_RVQ_B15_FIDELITY_CKPT")"
elif [[ "$START_PHASE" == "rvq_stage1" ]]; then
  if [[ "$RESUME_BYPASS_STAGE1" != "0" ]]; then
    echo "ERROR: RESUME_BYPASS_STAGE1 is not valid with START_PHASE=rvq_stage1." >&2
    exit 2
  fi
  if [[ -n "$PRETRAINED_Q0_CKPT" ]]; then
    if [[ ! -f "$PRETRAINED_Q0_CKPT" ]]; then
      echo "ERROR: PRETRAINED_Q0_CKPT does not exist: $PRETRAINED_Q0_CKPT" >&2
      exit 2
    fi
    RVQ_PROJECTION_CKPT="$(readlink -f "$PRETRAINED_Q0_CKPT")"
    B1_Q0_REUSE_FLAGS=(--reinitialize-rvq-codebooks-from-projection-checkpoint)
    BYPASS_S2_CKPT=""
  else
    if [[ -z "$BYPASS_STAGE2_CKPT" || ! -f "$BYPASS_STAGE2_CKPT" ]]; then
      echo "ERROR: START_PHASE=rvq_stage1 requires BYPASS_STAGE2_CKPT or PRETRAINED_Q0_CKPT." >&2
      echo "BYPASS_STAGE2_CKPT=${BYPASS_STAGE2_CKPT:-<empty>}" >&2
      echo "PRETRAINED_Q0_CKPT=${PRETRAINED_Q0_CKPT:-<empty>}" >&2
      exit 2
    fi
    BYPASS_S2_CKPT="$(readlink -f "$BYPASS_STAGE2_CKPT")"
    B1_Q0_REUSE_FLAGS=()
  fi
elif [[ "$START_PHASE" == "bypass_formant_refine" ]]; then
  if [[ "$RESUME_BYPASS_STAGE1" != "0" ]]; then
    echo "ERROR: RESUME_BYPASS_STAGE1 is not valid with START_PHASE=bypass_formant_refine." >&2
    exit 2
  fi
  if [[ -z "$BYPASS_STAGE1_CKPT" || ! -f "$BYPASS_STAGE1_CKPT" ]]; then
    echo "ERROR: START_PHASE=bypass_formant_refine requires an existing BYPASS_STAGE1_CKPT." >&2
    echo "received: ${BYPASS_STAGE1_CKPT:-<empty>}" >&2
    exit 2
  fi
  BYPASS_S1_CKPT="$(readlink -f "$BYPASS_STAGE1_CKPT")"
elif [[ "$RESUME_BYPASS_STAGE1" == "1" ]]; then
  if [[ ! -f "$BYPASS_S1_DIR/latest.pt" ]]; then
    echo "ERROR: resume requires $BYPASS_S1_DIR/latest.pt" >&2
    exit 2
  fi
  A1_RESUME_FLAGS=(--resume --reset-early-stopping-on-resume)
  A1_APPEND_LOG=1
else
  A1_RESUME_FLAGS=(--no-resume)
  A1_APPEND_LOG=0
  if [[ -e "$BYPASS_S1_DIR" ]]; then
    echo "ERROR: results directory already exists: $BYPASS_S1_DIR" >&2
    exit 2
  fi
fi

CHECK_DIRS=("$RVQ_PROJECTION_DIR" "$RVQ_S1_DIR" "$RVQ_B15_DIR" "$RVQ_B15_POLISH_DIR" "$RVQ_S2_DIR" "$FINAL_STATE_DIR")
if [[ -n "$PRETRAINED_Q0_CKPT" ]]; then
  CHECK_DIRS=("$RVQ_S1_DIR" "$RVQ_B15_DIR" "$RVQ_B15_POLISH_DIR" "$RVQ_S2_DIR" "$FINAL_STATE_DIR")
fi
if [[ "$START_PHASE" == "rvq_polish" ]]; then
  CHECK_DIRS=("$RVQ_B15_POLISH_DIR" "$RVQ_S2_DIR" "$FINAL_STATE_DIR")
fi
if [[ "$START_PHASE" == "bypass_stage1" || \
      "$START_PHASE" == "bypass_formant_refine" ]]; then
  CHECK_DIRS=("$BYPASS_FORMANT_DIR" "$BYPASS_S2_DIR" "${CHECK_DIRS[@]}")
  if [[ "$ENABLE_PRE_RVQ_STATE_FT" == "1" ]]; then
    CHECK_DIRS+=("$PRE_RVQ_STATE_DIR")
  fi
fi
for dir in "${CHECK_DIRS[@]}"; do
  if [ -e "$dir" ]; then
    echo "ERROR: results directory already exists: $dir" >&2
    echo "Use a new RUN_TAG. Existing training results are never overwritten." >&2
    exit 2
  fi
done

run_stage() {
  local label="$1" port="$2" log="$3" append_log="$4"
  shift 4
  echo "===== $label ====="
  echo "log=$log"
  if [[ "$append_log" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_LIST" accelerate launch \
      --multi_gpu --num_processes "$NUM_PROCESSES" --num_machines 1 \
      --main_process_port "$port" --dynamo_backend no --mixed_precision no \
      train_soundstream.py "$@" >> "$log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$GPU_LIST" accelerate launch \
      --multi_gpu --num_processes "$NUM_PROCESSES" --num_machines 1 \
      --main_process_port "$port" --dynamo_backend no --mixed_precision no \
      train_soundstream.py "$@" > "$log" 2>&1
  fi
}

pick_best() {
  local dir="$1" candidate
  for candidate in \
    best_rvq_projection.pt \
    best_by_formant.pt \
    best_full_gan_formant_balanced.pt \
    best_full_gan_balanced.pt \
    best_gan_balanced.pt \
    best_by_clarity.pt \
    best_selected.pt \
    best_by_aligned_si_sdr.pt; do
    if [ -f "$dir/$candidate" ]; then
      printf '%s\n' "$dir/$candidate"
      return 0
    fi
  done
  echo "ERROR: no validation-selected checkpoint in $dir" >&2
  return 1
}

pick_b15_best() {
  local dir="$1" candidate
  # B1.5 is a reconstruction handoff, so select the strongest aligned
  # waveform checkpoint before generic clarity/selection aliases.
  for candidate in \
    best_rvq_fidelity.pt \
    best_by_aligned_si_sdr.pt \
    best_selected.pt \
    best_by_clarity.pt; do
    if [ -f "$dir/$candidate" ]; then
      printf '%s\n' "$dir/$candidate"
      return 0
    fi
  done
  echo "ERROR: no validation-selected B1.5 checkpoint in $dir" >&2
  return 1
}

COMMON=(
  --audio-dir "$AUDIO_DIR" --batch-size 4 --grad-accum-every 1
  --num-quantizers "$RVQ_NUM_QUANTIZERS"
  --codebook-sizes "${RVQ_CODEBOOK_SIZE_VALUES[@]}"
  --rq-lookup-dim "$RVQ_LOOKUP_DIM"
  --rq-distance "$RVQ_DISTANCE"
  --segment-seconds 4 --dl-num-workers 6 --seed "$SEED"
  --decoder-upsample-mode convtranspose --no-decoder-split-first-upsample
  --save-model-every 5000 --test-eval-batches 100
)

RECON_LOSSES=(
  --si-sdr-loss-weight 0.07 --si-sdr-loss-start-steps 15000 --si-sdr-loss-warmup-steps 15000
  --spectral-envelope-loss-weight 0 --spectral-envelope-loss-start-steps 0 --spectral-envelope-loss-warmup-steps 0
  --formant-peak-loss-weight 0 --formant-peak-loss-start-steps 0 --formant-peak-loss-warmup-steps 0
  --stft-recon-loss-weight 0.05 --stft-recon-loss-start-steps 5000 --stft-recon-loss-warmup-steps 15000
  --voiced-highband-loss-weight 0 --voiced-highband-loss-start-steps 0 --voiced-highband-loss-warmup-steps 0
  --upper-highband-loss-weight 0 --upper-highband-loss-start-steps 0 --upper-highband-loss-warmup-steps 0
  --noise-floor-loss-weight 0.03 --click-loss-weight 0.002 --jump-loss-weight 0 --preemph-loss-weight 0
  --loss-grad-diagnostics-every 1000
)

GAN_LOSSES=(
  --generator-lr 5e-7 --gan-adversarial-max 2e-4 --gan-feature-max 1.0
  --si-sdr-loss-weight 0.05 --spectral-envelope-loss-weight 0.02 --formant-peak-loss-weight 0
  --stft-recon-loss-weight 0.02 --voiced-highband-loss-weight 0 --noise-floor-loss-weight 0.03
  --voiced-hf-retention-loss-weight 0.01 --upper-highband-loss-weight 0
  --active-spectral-detail-loss-weight 0
  --waveform-discr-lrs 5e-7 5e-7 2.5e-7 --stft-discr-lr 2.5e-7
  --waveform-discr-update-every 2 4 4 --waveform-discr-loss-weights 1.0 0.25 0.25
  --stft-discr-update-every 4 --stft-discr-loss-weight 0.5
  --stage2-generator-freeze-steps 2000 --stage2-generator-hold-steps 5000 --stage2-generator-hold-lr 1e-7
  --stage2-encoder-unfreeze-step 10000 --stage2-encoder-trainable-from-block 3 --stage2-encoder-lr 1e-7
  --stage2-phase2-start-step 2000 --stage2-phase3-start-step 10000
  --stage2-phase2-generator-lr 5e-7 --stage2-phase3-generator-lr 5e-7
  --early-stopping-min-steps 150000
  --no-stage2-quality-hard-stop --gan-grad-diagnostics-every 500 --loss-grad-diagnostics-every 1000
)

if [[ "$START_PHASE" != "rvq_stage1" && "$START_PHASE" != "rvq_polish" ]]; then
  if [[ "$START_PHASE" == "bypass_stage1" ]]; then
  # A1: RVQ is absent from both the forward signal and checkpoint selection gate.
  run_stage "A1 bypass-RVQ reconstruction" 29511 "$PWD/logs/bypass-recon-$BASE.log" "$A1_APPEND_LOG" \
    --stage recon_pretrain --results-dir "$BYPASS_S1_DIR" \
    --num-train-steps "$BYPASS_STAGE1_STEPS" --bypass-rvq-during-training \
    --early-stopping-min-steps "$BYPASS_STAGE1_EARLY_STOPPING_MIN_STEPS" \
    --preceding-context-seconds "$PRECEDING_CONTEXT_SECONDS" \
    --decoder-residual-scale-start 0.2 --decoder-residual-scale-end 1.0 \
    --decoder-residual-scale-warmup-start-steps 0 --decoder-residual-scale-warmup-end-steps 15000 \
    "${RECON_LOSSES[@]}" "${COMMON[@]}" "${A1_RESUME_FLAGS[@]}"
  BYPASS_S1_CKPT="$(pick_best "$BYPASS_S1_DIR")"
  echo "A1 checkpoint=$BYPASS_S1_CKPT"
  if [[ "$STOP_AFTER_PHASE" == "bypass_stage1" ]]; then
    echo "Stopping after A1 as requested."
    exit 0
  fi
  else
    echo "Skipping A1; starting A1.5 from $BYPASS_S1_CKPT"
  fi

  # A1.5: keep RVQ bypassed, update Decoder at 1e-5 and Encoder at 2e-6,
  # and select a formant-preserving reconstruction before adversarial training.
  run_stage "A1.5 bypass-RVQ formant refinement" 29515 \
    "$PWD/logs/bypass-formant-refine-$BASE.log" 0 \
    --stage spectral_refine --results-dir "$BYPASS_FORMANT_DIR" \
    --init-checkpoint "$BYPASS_S1_CKPT" \
    --num-train-steps "$BYPASS_FORMANT_REFINE_STEPS" \
    --bypass-rvq-during-training \
    --preceding-context-seconds "$PRECEDING_CONTEXT_SECONDS" \
    --spectral-envelope-loss-weight 0.10 \
    --spectral-envelope-loss-start-steps 0 \
    --spectral-envelope-loss-warmup-steps 0 \
    --formant-peak-loss-weight 0.05 \
    --formant-peak-loss-start-steps 0 \
    --formant-peak-loss-warmup-steps 5000 \
    --voiced-highband-loss-weight 0.02 \
    --voiced-highband-loss-start-steps 0 \
    --voiced-highband-loss-warmup-steps 5000 \
    --upper-highband-loss-weight 0 \
    --stft-recon-loss-weight 0.05 \
    --stft-recon-loss-start-steps 0 \
    --stft-recon-loss-warmup-steps 5000 \
    --si-sdr-loss-weight 0.05 \
    --noise-floor-loss-weight 0.03 \
    "${COMMON[@]}" --no-resume
  if [[ -f "$BYPASS_FORMANT_DIR/best_by_formant.pt" ]]; then
    BYPASS_FORMANT_CKPT="$BYPASS_FORMANT_DIR/best_by_formant.pt"
  elif [[ -f "$BYPASS_FORMANT_DIR/baseline_init.pt" ]]; then
    BYPASS_FORMANT_CKPT="$BYPASS_FORMANT_DIR/baseline_init.pt"
  else
    echo "ERROR: A1.5 produced neither an improved formant checkpoint nor its baseline." >&2
    exit 1
  fi
  echo "A1.5 checkpoint=$BYPASS_FORMANT_CKPT"
  if [[ "$STOP_AFTER_PHASE" == "bypass_formant_refine" ]]; then
    echo "Stopping after A1.5 as requested."
    exit 0
  fi

  # A2: warm up discriminators, train Decoder from 2k, then jointly train the
  # Encoder Block4/final latent convolution and Decoder from 10k. RVQ remains
  # bypassed and frozen throughout this phase.
  run_stage "A2 bypass-RVQ GAN" 29512 "$PWD/logs/bypass-gan-$BASE.log" 0 \
    --stage gan_pretrain --results-dir "$BYPASS_S2_DIR" \
    --init-checkpoint "$BYPASS_FORMANT_CKPT" --num-train-steps "$BYPASS_STAGE2_STEPS" \
    --bypass-rvq-during-training \
    --preceding-context-seconds "$PRECEDING_CONTEXT_SECONDS" \
    "${GAN_LOSSES[@]}" "${COMMON[@]}" --no-resume
  BYPASS_S2_CKPT="$(pick_best "$BYPASS_S2_DIR")"
  echo "A2 checkpoint=$BYPASS_S2_CKPT"
  if [[ "$STOP_AFTER_PHASE" == "bypass_stage2" ]]; then
    echo "Stopping after bypass A2 as requested."
    exit 0
  fi
  if [[ "$ENABLE_PRE_RVQ_STATE_FT" == "1" ]]; then
    run_stage "A2.5 pre-RVQ stateful alignment" 29516 \
      "$PWD/logs/state-pre-rvq-$BASE.log" 0 \
      --stage stream_finetune --results-dir "$PRE_RVQ_STATE_DIR" \
      --init-checkpoint "$BYPASS_S2_CKPT" \
      --num-train-steps "$PRE_RVQ_STATE_STEPS" \
      --bypass-rvq-during-training \
      --preceding-context-seconds "$PRECEDING_CONTEXT_SECONDS" \
      --stream-frame-size 320 --stream-tbptt-frames 20 \
      "${COMMON[@]}" --no-resume
    BYPASS_S2_CKPT="$(pick_best "$PRE_RVQ_STATE_DIR")"
    echo "A2.5 state-aligned checkpoint=$BYPASS_S2_CKPT"
  else
    echo "A2.5 pre-RVQ stateful alignment skipped; run the consistency check before B1."
  fi
else
  if [[ "$START_PHASE" == "rvq_polish" ]]; then
    echo "Skipping Q0, B1, and B1.5a; resuming from validation-selected RVQ fidelity checkpoint: $RVQ_B15_FIDELITY_CKPT"
  elif [[ -n "$PRETRAINED_Q0_CKPT" ]]; then
    echo "Skipping bypass and Q0 training; reusing validated Q0 checkpoint: $RVQ_PROJECTION_CKPT"
    echo "Q0 reuse gain: preserves trained 64<->${RVQ_LOOKUP_DIM} projections and avoids up to ${RVQ_PROJECTION_STEPS} projection-training steps."
  else
    echo "Skipping bypass stages; starting RVQ Stage-1 from $BYPASS_S2_CKPT"
  fi
fi

echo "State policy: pre-RVQ A2.5 enabled=$ENABLE_PRE_RVQ_STATE_FT; post-RVQ state fine-tune is mandatory (${FINAL_STATE_STEPS} steps)."

# Q0: estimate the frozen Encoder latent rank across all workers, initialize
# Win/Wout from centered PCA, and train only the two projections through the
# frozen Decoder. A validation-selected Q0 may be reused explicitly; B1 then
# retains Win/Wout and rebuilds only rq.* codebooks and EMA state.  A failed
# B1.5b may resume explicitly from B1.5a's validation-selected fidelity model.
if [[ "$START_PHASE" != "rvq_polish" ]]; then
if [[ -z "$PRETRAINED_Q0_CKPT" ]]; then
run_stage "Q0 PCA projection-only pretraining" 29519 \
  "$PWD/logs/rvq-projection-$BASE.log" 0 \
  --stage recon_pretrain --results-dir "$RVQ_PROJECTION_DIR" \
  --init-checkpoint "$BYPASS_S2_CKPT" --num-train-steps "$RVQ_PROJECTION_STEPS" \
  --reinitialize-rvq-from-bypass-checkpoint \
  --rvq-projection-only \
  --rvq-projection-pca-batches "$RVQ_PROJECTION_PCA_BATCHES" \
  --rvq-projection-min-evr "$RVQ_PROJECTION_MIN_EVR" \
  --rvq-projection-latent-mse-weight "$RVQ_PROJECTION_LATENT_MSE_WEIGHT" \
  --rvq-projection-latent-cosine-weight "$RVQ_PROJECTION_LATENT_COSINE_WEIGHT" \
  --rvq-projection-orth-loss-weight "$RVQ_PROJECTION_ORTH_WEIGHT" \
  --rvq-projection-tie-loss-weight "$RVQ_PROJECTION_TIE_WEIGHT" \
  --rvq-projection-freeze-input-steps "$RVQ_PROJECTION_FREEZE_INPUT_STEPS" \
  --clean-gate-min-aligned-si-sdr "$RVQ_PROJECTION_MIN_ALIGNED_SI_SDR" \
  "${RECON_LOSSES[@]}" "${COMMON[@]}" \
  --generator-lr "$RVQ_PROJECTION_LR" \
  --si-sdr-loss-weight 0.07 --si-sdr-loss-start-steps 0 \
  --si-sdr-loss-warmup-steps 2000 \
  --stft-recon-loss-weight 0.05 --stft-recon-loss-start-steps 0 \
  --stft-recon-loss-warmup-steps 2500 \
  --early-stopping-min-steps "$RVQ_PROJECTION_EARLY_STOPPING_MIN_STEPS" \
  --early-stopping-patience "$RVQ_PROJECTION_EARLY_STOPPING_PATIENCE" \
  --no-bypass-rvq-during-training --no-resume
RVQ_PROJECTION_CKPT="$(pick_best "$RVQ_PROJECTION_DIR" 2>/dev/null || true)"
if [[ -z "$RVQ_PROJECTION_CKPT" ]]; then
  echo "ERROR: Q0 projection path produced no clean validation-selected checkpoint." >&2
  echo "Inspect $RVQ_PROJECTION_DIR/latent_pca_report.json and test lookup dim 24 or 32." >&2
  exit 2
fi
echo "Q0 projection checkpoint=$RVQ_PROJECTION_CKPT"
else
  echo "Reused Q0 projection checkpoint=$RVQ_PROJECTION_CKPT"
fi

# B1: initialize every level by sequential residual K-means from a large
# projected-latent pool, then run the short EMA/dead-code bootstrap. There is
# no Decoder call, backward pass, or optimizer update in this stage.
run_stage "B1 RVQ calibration with codec frozen" 29513 "$PWD/logs/rvq-calibration-$BASE.log" 0 \
  --stage recon_pretrain --results-dir "$RVQ_S1_DIR" \
  --init-checkpoint "$RVQ_PROJECTION_CKPT" --num-train-steps "$RVQ_CALIBRATION_STEPS" \
  --rvq-calibration-only \
  --rvq-calibration-kmeans-batches "$RVQ_CALIBRATION_KMEANS_BATCHES" \
  --early-stopping-min-steps 0 \
  --no-bypass-rvq-during-training "${B1_Q0_REUSE_FLAGS[@]}" "${COMMON[@]}" --no-resume
RVQ_S1_CKPT="$RVQ_S1_DIR/latest.pt"
test -f "$RVQ_S1_CKPT" || { echo "ERROR: missing $RVQ_S1_CKPT" >&2; exit 2; }
echo "B1 checkpoint=$RVQ_S1_CKPT"

# B1.5a: keep Encoder, Win and Wout immutable. Adapt Decoder first, then
# allow conservative assignment-driven RVQ EMA updates.  The trainer records
# a dedicated fidelity checkpoint using aligned SI-SDR with lookup/64-D NMSE.
run_stage "B1.5a joint STE quantization adaptation" 29518 \
  "$PWD/logs/rvq-joint-adapt-$BASE.log" 0 \
  --stage gan_pretrain --results-dir "$RVQ_B15_DIR" \
  --init-checkpoint "$RVQ_S1_CKPT" --num-train-steps "$RVQ_B15_RVQ_ADAPT_END_STEPS" \
  "${RECON_LOSSES[@]}" "${COMMON[@]}" \
  --rvq-joint-adapt --rvq-warm-in-steps 5000 \
  --rvq-joint-decoder-only-steps "$RVQ_B15_DECODER_ONLY_STEPS" \
  --rvq-joint-rvq-adapt-end-steps "$RVQ_B15_RVQ_ADAPT_END_STEPS" \
  --rvq-joint-polish-decoder-lr "$RVQ_B15_POLISH_DECODER_LR" \
  --rvq-continuous-teacher-loss-weight 0.20 \
  --rvq-joint-latent64-teacher-loss-weight 0 \
  --si-sdr-loss-weight 0.10 --si-sdr-loss-start-steps 0 \
  --si-sdr-loss-warmup-steps 2500 \
  --stft-recon-loss-weight 0.05 --stft-recon-loss-start-steps 0 \
  --stft-recon-loss-warmup-steps 2500 \
  --active-spectral-detail-loss-weight 0 \
  --voiced-highband-loss-weight 0 \
  --voiced-hf-retention-loss-weight 0 \
  --upper-highband-loss-weight 0 \
  --spectral-envelope-loss-weight 0 \
  --formant-peak-loss-weight 0 \
  --clean-gate-min-aligned-si-sdr "$RVQ_B15_MIN_ALIGNED_SI_SDR" \
  --clean-gate-max-negative-fraction 0.05 \
  --no-bypass-rvq-during-training --no-resume
RVQ_B15_FIDELITY_CKPT="$RVQ_B15_DIR/best_rvq_fidelity.pt"
test -f "$RVQ_B15_FIDELITY_CKPT" || {
  echo "ERROR: B1.5a produced no best_rvq_fidelity.pt during the RVQ EMA phase." >&2
  exit 2
}
echo "B1.5a fidelity checkpoint=$RVQ_B15_FIDELITY_CKPT"
else
  echo "Reused B1.5a fidelity checkpoint=$RVQ_B15_FIDELITY_CKPT"
fi

# B1.5b: restore the best EMA-phase fidelity point, freeze RVQ for every
# polish step, and use the reduced Decoder LR.  Together B1.5a+B1.5b consume
# RVQ_JOINT_ADAPT_STEPS (default 45k + 15k = 60k).
run_stage "B1.5b Decoder polish from best RVQ fidelity" 29519 \
  "$PWD/logs/rvq-decoder-polish-$BASE.log" 0 \
  --stage gan_pretrain --results-dir "$RVQ_B15_POLISH_DIR" \
  --init-checkpoint "$RVQ_B15_FIDELITY_CKPT" \
  --num-train-steps "$RVQ_B15_POLISH_STEPS" \
  --early-stopping-min-steps "$RVQ_B15_POLISH_STEPS" \
  "${RECON_LOSSES[@]}" "${COMMON[@]}" \
  --generator-lr "$RVQ_B15_POLISH_DECODER_LR" \
  --rvq-joint-adapt --rvq-warm-in-steps 0 \
  --rvq-joint-decoder-only-steps "$RVQ_B15_POLISH_STEPS" \
  --rvq-joint-rvq-adapt-end-steps "$RVQ_B15_POLISH_STEPS" \
  --rvq-joint-polish-decoder-lr "$RVQ_B15_POLISH_DECODER_LR" \
  --rvq-continuous-teacher-loss-weight 0.20 \
  --rvq-joint-latent64-teacher-loss-weight 0 \
  --si-sdr-loss-weight 0.10 --si-sdr-loss-start-steps 0 \
  --si-sdr-loss-warmup-steps 2500 \
  --stft-recon-loss-weight 0.05 --stft-recon-loss-start-steps 0 \
  --stft-recon-loss-warmup-steps 2500 \
  --active-spectral-detail-loss-weight 0 \
  --voiced-highband-loss-weight 0 \
  --voiced-hf-retention-loss-weight 0 \
  --upper-highband-loss-weight 0 \
  --spectral-envelope-loss-weight 0 \
  --formant-peak-loss-weight 0 \
  --clean-gate-min-aligned-si-sdr "$RVQ_B15_MIN_ALIGNED_SI_SDR" \
  --clean-gate-max-negative-fraction 0.05 \
  --no-bypass-rvq-during-training --no-resume
RVQ_B15_CKPT="$(pick_b15_best "$RVQ_B15_POLISH_DIR" 2>/dev/null || true)"
test -f "$RVQ_B15_CKPT" || {
  echo "ERROR: B1.5b produced no validation-selected checkpoint; latest.pt is resume-only." >&2
  exit 2
}
echo "B1.5 final checkpoint=$RVQ_B15_CKPT"

# The fidelity gate belongs after joint representation learning, not after the
# short B1 codebook bootstrap. GAN must not be used to hide RVQ distortion.
B15_VALIDATION_REPORT="$RVQ_B15_POLISH_DIR/b15_fixed_validation.tsv"
run_stage "B1.5 fixed quantization-fidelity validation" 29520 \
  "$PWD/logs/rvq-joint-adapt-validation-$BASE.log" 0 \
  --stage gan_pretrain --results-dir "$RVQ_B15_POLISH_DIR" \
  --validation-only --validation-checkpoint "$RVQ_B15_CKPT" \
  --validation-report-file "$B15_VALIDATION_REPORT" \
  --no-bypass-rvq-during-training "${COMMON[@]}" --no-resume
python - \
  "$B15_VALIDATION_REPORT" \
  "$RVQ_B1_MIN_ALIGNED_SI_SDR" \
  "$RVQ_B1_MIN_QUANTIZATION_GAP_DB" \
  "$RVQ_B15_MAX_LOOKUP_NMSE" \
  "$RVQ_B15_MAX_LATENT64_NMSE" <<'PY'
import sys
from pathlib import Path

report = Path(sys.argv[1])
minimum_si_sdr = float(sys.argv[2])
minimum_gap = float(sys.argv[3])
maximum_lookup_nmse = float(sys.argv[4])
maximum_latent64_nmse = float(sys.argv[5])
metrics = {}
with report.open("r", encoding="utf-8") as handle:
    header = handle.readline().rstrip("\n").split("\t")
    if header != ["metric", "value"]:
        raise SystemExit(f"RVQ handoff FAILED: unexpected report header {header}")
    for line in handle:
        name, value = line.rstrip("\n").split("\t", 1)
        metrics[name] = float(value)

required = (
    "aligned_si_sdr",
    "projection_only_aligned_si_sdr",
    "quantization_gap_db",
    "rvq_latent_nmse_lookup",
    "rvq_latent_nmse_64d",
)
missing = [name for name in required if name not in metrics]
if missing:
    raise SystemExit("RVQ handoff FAILED: missing metrics: " + ", ".join(missing))

aligned = metrics["aligned_si_sdr"]
projection = metrics["projection_only_aligned_si_sdr"]
gap = metrics["quantization_gap_db"]
lookup_nmse = metrics["rvq_latent_nmse_lookup"]
latent64_nmse = metrics["rvq_latent_nmse_64d"]
print(
    "RVQ handoff: "
    f"projection_aligned_si_sdr={projection:.3f} dB, "
    f"quantized_aligned_si_sdr={aligned:.3f} dB, "
    f"gap={gap:+.3f} dB, lookup_nmse={lookup_nmse:.4f}, "
    f"latent64_nmse={latent64_nmse:.4f}"
)
failures = []
if aligned < minimum_si_sdr:
    failures.append(f"aligned_si_sdr {aligned:.3f} < {minimum_si_sdr:.3f}")
if gap < minimum_gap:
    failures.append(f"quantization_gap_db {gap:.3f} < {minimum_gap:.3f}")
if lookup_nmse > maximum_lookup_nmse:
    failures.append(f"lookup_nmse {lookup_nmse:.4f} > {maximum_lookup_nmse:.4f}")
if latent64_nmse > maximum_latent64_nmse:
    failures.append(
        f"latent64_nmse {latent64_nmse:.4f} > {maximum_latent64_nmse:.4f}"
    )
if failures:
    raise SystemExit("RVQ handoff FAILED: " + "; ".join(failures))
print("RVQ handoff PASSED")
PY

# B2: Encoder and RVQ are frozen for the whole shortened GAN phase;
# Decoder adapts to quantization error while GAN discriminators are trained.
run_stage "B2 RVQ GAN decoder adaptation" 29514 "$PWD/logs/rvq-gan-$BASE.log" 0 \
  --stage gan_pretrain --results-dir "$RVQ_S2_DIR" \
  --init-checkpoint "$RVQ_B15_CKPT" --num-train-steps "$RVQ_STAGE2_STEPS" \
  --no-bypass-rvq-during-training "${GAN_LOSSES[@]}" \
  --stage2-encoder-unfreeze-step -1 --early-stopping-min-steps 10000 \
  --early-stopping-patience 20 --stage2-quality-hard-stop \
  --clean-gate-max-negative-fraction 0.05 \
  "${COMMON[@]}" --no-resume
RVQ_S2_CKPT="$(pick_best "$RVQ_S2_DIR")"
echo "B2 checkpoint=$RVQ_S2_CKPT"

# Final state alignment is mandatory for the deployable path. The
# stream_finetune_long policy freezes Encoder and RVQ for the complete stage,
# keeps GAN losses disabled, and updates only Decoder parameters against
# persistent 20 ms state / boundary consistency losses.
run_stage "Final post-RVQ stateful Decoder fine-tune" 29517 \
  "$PWD/logs/state-final-rvq-$BASE.log" 0 \
  --stage stream_finetune_long --results-dir "$FINAL_STATE_DIR" \
  --init-checkpoint "$RVQ_S2_CKPT" \
  --num-train-steps "$FINAL_STATE_STEPS" \
  --no-bypass-rvq-during-training \
  --preceding-context-seconds "$PRECEDING_CONTEXT_SECONDS" \
  --stream-frame-size 320 --stream-tbptt-frames 20 \
  "${COMMON[@]}" --no-resume
FINAL_STATE_CKPT="$(pick_best "$FINAL_STATE_DIR")"
echo "Final stateful checkpoint=$FINAL_STATE_CKPT"

echo "===== deployment training pipeline complete ====="
echo "bypass Stage-1: $BYPASS_S1_DIR"
echo "bypass formant refine: $BYPASS_FORMANT_DIR"
echo "bypass Stage-2: $BYPASS_S2_DIR"
if [[ "$ENABLE_PRE_RVQ_STATE_FT" == "1" ]]; then
  echo "pre-RVQ stateful fine-tune: $PRE_RVQ_STATE_DIR"
fi
if [[ -n "$PRETRAINED_Q0_CKPT" ]]; then
  echo "RVQ projection reused: $RVQ_PROJECTION_CKPT"
else
  echo "RVQ projection pretraining: $RVQ_PROJECTION_DIR"
fi
echo "RVQ calibration: $RVQ_S1_DIR"
echo "RVQ joint adaptation: $RVQ_B15_DIR"
echo "RVQ fidelity-restored Decoder polish: $RVQ_B15_POLISH_DIR"
echo "RVQ Stage-2: $RVQ_S2_DIR"
echo "Final stateful fine-tune: $FINAL_STATE_DIR"
