#!/usr/bin/env bash
set -euo pipefail

# Deployment-oriented training:
#   A1 Encoder -> Decoder reconstruction (RVQ bypassed)
#   A1.5 low-LR Encoder/Decoder formant refinement (RVQ bypassed)
#   A2 Encoder -> Decoder GAN refinement (RVQ bypassed)
#   optional A2.5 persistent-state alignment before RVQ calibration
#   Q0 PCA initialization + projection-only reconstruction pretraining
#   B1 Encoder -> RVQ EMA calibration (Encoder/Decoder weights frozen)
#   B1.5 non-GAN STE adaptation (Decoder + Encoder tail + RVQ EMA schedule)
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
BYPASS_STAGE1_STEPS="${BYPASS_STAGE1_STEPS:-150000}"
BYPASS_STAGE1_EARLY_STOPPING_MIN_STEPS="${BYPASS_STAGE1_EARLY_STOPPING_MIN_STEPS:-60000}"
BYPASS_FORMANT_REFINE_STEPS="${BYPASS_FORMANT_REFINE_STEPS:-20000}"
BYPASS_STAGE2_STEPS="${BYPASS_STAGE2_STEPS:-150000}"
RVQ_PROJECTION_STEPS="${RVQ_PROJECTION_STEPS:-10000}"
RVQ_PROJECTION_PCA_BATCHES="${RVQ_PROJECTION_PCA_BATCHES:-100}"
RVQ_PROJECTION_MIN_EVR="${RVQ_PROJECTION_MIN_EVR:-0.90}"
RVQ_PROJECTION_MIN_ALIGNED_SI_SDR="${RVQ_PROJECTION_MIN_ALIGNED_SI_SDR:-7.3}"
RVQ_PROJECTION_LATENT_MSE_WEIGHT="${RVQ_PROJECTION_LATENT_MSE_WEIGHT:-0.25}"
RVQ_PROJECTION_LATENT_COSINE_WEIGHT="${RVQ_PROJECTION_LATENT_COSINE_WEIGHT:-0.10}"
RVQ_CALIBRATION_STEPS="${RVQ_CALIBRATION_STEPS:-5000}"
RVQ_JOINT_ADAPT_STEPS="${RVQ_JOINT_ADAPT_STEPS:-20000}"
RVQ_STAGE2_STEPS="${RVQ_STAGE2_STEPS:-50000}"
ENABLE_PRE_RVQ_STATE_FT="${ENABLE_PRE_RVQ_STATE_FT:-0}"
PRE_RVQ_STATE_STEPS="${PRE_RVQ_STATE_STEPS:-20000}"
FINAL_STATE_STEPS="${FINAL_STATE_STEPS:-10000}"
RESUME_BYPASS_STAGE1="${RESUME_BYPASS_STAGE1:-0}"
START_PHASE="${START_PHASE:-bypass_stage1}"
STOP_AFTER_PHASE="${STOP_AFTER_PHASE:-}"
BYPASS_STAGE1_CKPT="${BYPASS_STAGE1_CKPT:-}"
BYPASS_STAGE2_CKPT="${BYPASS_STAGE2_CKPT:-}"
PRECEDING_CONTEXT_SECONDS="${PRECEDING_CONTEXT_SECONDS:-0.5}"

case "$RVQ_LOOKUP_DIM" in
  16|24|32) ;;
  *) echo "ERROR: RVQ_LOOKUP_DIM must be 16, 24, or 32." >&2; exit 2 ;;
esac

if (( BYPASS_STAGE1_EARLY_STOPPING_MIN_STEPS > BYPASS_STAGE1_STEPS )); then
  BYPASS_STAGE1_EARLY_STOPPING_MIN_STEPS="$BYPASS_STAGE1_STEPS"
fi

BASE="lowrank-dscnn-b234-relu-convtranspose-fp-64d-8q256-l${RVQ_LOOKUP_DIM}-${RUN_TAG}"
BYPASS_S1_DIR="$PWD/results/bypass-recon-$BASE"
BYPASS_FORMANT_DIR="$PWD/results/bypass-formant-refine-$BASE"
BYPASS_S2_DIR="$PWD/results/bypass-gan-$BASE"
RVQ_PROJECTION_DIR="$PWD/results/rvq-projection-$BASE"
RVQ_S1_DIR="$PWD/results/rvq-calibration-$BASE"
RVQ_B15_DIR="$PWD/results/rvq-joint-adapt-$BASE"
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
if (( RVQ_JOINT_ADAPT_STEPS < 15000 )); then
  echo "ERROR: RVQ_JOINT_ADAPT_STEPS must be at least 15000 for all B1.5 phases." >&2
  exit 2
fi
if (( RVQ_STAGE2_STEPS <= 0 )); then
  echo "ERROR: RVQ_STAGE2_STEPS must be positive." >&2
  exit 2
fi

if [[ "$START_PHASE" != "bypass_stage1" && \
      "$START_PHASE" != "bypass_formant_refine" && \
      "$START_PHASE" != "rvq_stage1" ]]; then
  echo "ERROR: START_PHASE must be bypass_stage1, bypass_formant_refine, or rvq_stage1." >&2
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

if [[ "$START_PHASE" == "rvq_stage1" ]]; then
  if [[ "$RESUME_BYPASS_STAGE1" != "0" ]]; then
    echo "ERROR: RESUME_BYPASS_STAGE1 is not valid with START_PHASE=rvq_stage1." >&2
    exit 2
  fi
  if [[ -z "$BYPASS_STAGE2_CKPT" || ! -f "$BYPASS_STAGE2_CKPT" ]]; then
    echo "ERROR: START_PHASE=rvq_stage1 requires an existing BYPASS_STAGE2_CKPT." >&2
    echo "received: ${BYPASS_STAGE2_CKPT:-<empty>}" >&2
    exit 2
  fi
  BYPASS_S2_CKPT="$BYPASS_STAGE2_CKPT"
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

CHECK_DIRS=("$RVQ_PROJECTION_DIR" "$RVQ_S1_DIR" "$RVQ_B15_DIR" "$RVQ_S2_DIR" "$FINAL_STATE_DIR")
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

COMMON=(
  --audio-dir "$AUDIO_DIR" --batch-size 4 --grad-accum-every 1
  --rq-lookup-dim "$RVQ_LOOKUP_DIM"
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

if [[ "$START_PHASE" != "rvq_stage1" ]]; then
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
  echo "Skipping bypass stages; starting RVQ Stage-1 from $BYPASS_S2_CKPT"
fi

echo "State policy: pre-RVQ A2.5 enabled=$ENABLE_PRE_RVQ_STATE_FT; post-RVQ state fine-tune is mandatory (${FINAL_STATE_STEPS} steps)."

# Q0: estimate the frozen Encoder latent rank across all workers, initialize
# Win/Wout from centered PCA, and train only the two projections through the
# frozen Decoder.  Do not enter B1 unless reconstruction selection succeeds.
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
  --clean-gate-min-aligned-si-sdr "$RVQ_PROJECTION_MIN_ALIGNED_SI_SDR" \
  --generator-lr 1e-4 --early-stopping-min-steps "$RVQ_PROJECTION_STEPS" \
  --no-bypass-rvq-during-training "${RECON_LOSSES[@]}" "${COMMON[@]}" --no-resume
RVQ_PROJECTION_CKPT="$(pick_best "$RVQ_PROJECTION_DIR" 2>/dev/null || true)"
if [[ -z "$RVQ_PROJECTION_CKPT" ]]; then
  echo "ERROR: Q0 projection path produced no clean validation-selected checkpoint." >&2
  echo "Inspect $RVQ_PROJECTION_DIR/latent_pca_report.json and test lookup dim 24 or 32." >&2
  exit 2
fi
echo "Q0 projection checkpoint=$RVQ_PROJECTION_CKPT"

# B1: no Decoder call and no optimizer/backward step.  Only RVQ EMA and
# dead-code replacement state changes; Encoder/Decoder tensors remain bitwise fixed.
run_stage "B1 RVQ calibration with codec frozen" 29513 "$PWD/logs/rvq-calibration-$BASE.log" 0 \
  --stage recon_pretrain --results-dir "$RVQ_S1_DIR" \
  --init-checkpoint "$RVQ_PROJECTION_CKPT" --num-train-steps "$RVQ_CALIBRATION_STEPS" \
  --rvq-calibration-only --early-stopping-min-steps 0 \
  --no-bypass-rvq-during-training "${COMMON[@]}" --no-resume
RVQ_S1_CKPT="$RVQ_S1_DIR/latest.pt"
test -f "$RVQ_S1_CKPT" || { echo "ERROR: missing $RVQ_S1_CKPT" >&2; exit 2; }
echo "B1 checkpoint=$RVQ_S1_CKPT"

# B1.5: non-adversarial quantization adaptation.  The first 2k steps train
# Decoder only while alpha ramps z -> z_q over 10k steps. From 2k to 15k only
# Encoder Block4/final latent conv joins Decoder while RVQ EMA follows the
# changing latent distribution.  At 15k Encoder and RVQ refreeze.
run_stage "B1.5 joint STE quantization adaptation" 29518 \
  "$PWD/logs/rvq-joint-adapt-$BASE.log" 0 \
  --stage gan_pretrain --results-dir "$RVQ_B15_DIR" \
  --init-checkpoint "$RVQ_S1_CKPT" --num-train-steps "$RVQ_JOINT_ADAPT_STEPS" \
  --rvq-joint-adapt --rvq-warm-in-steps 10000 \
  --rvq-codebook-balance-loss-weight 0.01 \
  --rvq-codebook-balance-target-perplexity 64 \
  --rvq-codebook-balance-temperature 0.1 \
  --no-bypass-rvq-during-training "${RECON_LOSSES[@]}" "${COMMON[@]}" --no-resume
RVQ_B15_CKPT="$(pick_best "$RVQ_B15_DIR" 2>/dev/null || true)"
if [[ -z "$RVQ_B15_CKPT" && -f "$RVQ_B15_DIR/latest.pt" ]]; then
  RVQ_B15_CKPT="$RVQ_B15_DIR/latest.pt"
fi
test -f "$RVQ_B15_CKPT" || { echo "ERROR: B1.5 produced no checkpoint" >&2; exit 2; }
echo "B1.5 checkpoint=$RVQ_B15_CKPT"

# B2: Encoder and RVQ are frozen for the whole shortened GAN phase;
# Decoder adapts to quantization error while GAN discriminators are trained.
run_stage "B2 RVQ GAN decoder adaptation" 29514 "$PWD/logs/rvq-gan-$BASE.log" 0 \
  --stage gan_pretrain --results-dir "$RVQ_S2_DIR" \
  --init-checkpoint "$RVQ_B15_CKPT" --num-train-steps "$RVQ_STAGE2_STEPS" \
  --no-bypass-rvq-during-training "${GAN_LOSSES[@]}" \
  --stage2-encoder-unfreeze-step -1 --early-stopping-min-steps 10000 \
  --early-stopping-patience 20 --stage2-quality-hard-stop \
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
echo "RVQ projection pretraining: $RVQ_PROJECTION_DIR"
echo "RVQ calibration: $RVQ_S1_DIR"
echo "RVQ joint adaptation: $RVQ_B15_DIR"
echo "RVQ Stage-2: $RVQ_S2_DIR"
echo "Final stateful fine-tune: $FINAL_STATE_DIR"
