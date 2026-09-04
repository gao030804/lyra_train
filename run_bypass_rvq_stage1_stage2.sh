#!/usr/bin/env bash
set -euo pipefail

# Four-phase training:
#   A1 Encoder -> Decoder reconstruction (RVQ bypassed)
#   A2 Encoder -> Decoder GAN refinement (RVQ bypassed)
#   B1 Encoder -> RVQ EMA calibration (Encoder/Decoder weights frozen)
#   B2 Encoder -> frozen RVQ -> Decoder GAN adaptation (Encoder/RVQ frozen)

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
BYPASS_STAGE1_STEPS="${BYPASS_STAGE1_STEPS:-150000}"
BYPASS_STAGE2_STEPS="${BYPASS_STAGE2_STEPS:-150000}"
RVQ_CALIBRATION_STEPS="${RVQ_CALIBRATION_STEPS:-5000}"
RVQ_STAGE2_STEPS="${RVQ_STAGE2_STEPS:-150000}"
RESUME_BYPASS_STAGE1="${RESUME_BYPASS_STAGE1:-0}"

BASE="lowrank-dscnn-b234-relu-convtranspose-fp-64d-16q16-${RUN_TAG}"
BYPASS_S1_DIR="$PWD/results/bypass-recon-$BASE"
BYPASS_S2_DIR="$PWD/results/bypass-gan-$BASE"
RVQ_S1_DIR="$PWD/results/rvq-calibration-$BASE"
RVQ_S2_DIR="$PWD/results/rvq-gan-$BASE"

if [[ "$RESUME_BYPASS_STAGE1" != "0" && "$RESUME_BYPASS_STAGE1" != "1" ]]; then
  echo "ERROR: RESUME_BYPASS_STAGE1 must be 0 or 1." >&2
  exit 2
fi

if [[ "$RESUME_BYPASS_STAGE1" == "1" ]]; then
  if [[ ! -f "$BYPASS_S1_DIR/latest.pt" ]]; then
    echo "ERROR: resume requires $BYPASS_S1_DIR/latest.pt" >&2
    exit 2
  fi
  A1_RESUME_FLAG="--resume"
  A1_APPEND_LOG=1
else
  A1_RESUME_FLAG="--no-resume"
  A1_APPEND_LOG=0
  if [[ -e "$BYPASS_S1_DIR" ]]; then
    echo "ERROR: results directory already exists: $BYPASS_S1_DIR" >&2
    exit 2
  fi
fi

for dir in "$BYPASS_S2_DIR" "$RVQ_S1_DIR" "$RVQ_S2_DIR"; do
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
  --segment-seconds 4 --dl-num-workers 6 --seed "$SEED"
  --decoder-upsample-mode convtranspose --no-decoder-split-first-upsample
  --save-model-every 5000 --test-eval-batches 100
)

RECON_LOSSES=(
  --si-sdr-loss-weight 0.07 --si-sdr-loss-start-steps 5000 --si-sdr-loss-warmup-steps 15000
  --spectral-envelope-loss-weight 0.08 --spectral-envelope-loss-start-steps 5000 --spectral-envelope-loss-warmup-steps 15000
  --formant-peak-loss-weight 0.02 --formant-peak-loss-start-steps 15000 --formant-peak-loss-warmup-steps 20000
  --stft-recon-loss-weight 0.05 --stft-recon-loss-start-steps 5000 --stft-recon-loss-warmup-steps 15000
  --voiced-highband-loss-weight 0.04 --voiced-highband-loss-start-steps 5000 --voiced-highband-loss-warmup-steps 15000
  --noise-floor-loss-weight 0.03 --click-loss-weight 0.002 --jump-loss-weight 0 --preemph-loss-weight 0
  --loss-grad-diagnostics-every 1000
)

GAN_LOSSES=(
  --generator-lr 5e-7 --gan-adversarial-max 2e-4 --gan-feature-max 1.5
  --si-sdr-loss-weight 0.05 --spectral-envelope-loss-weight 0.05 --formant-peak-loss-weight 0.01
  --stft-recon-loss-weight 0.02 --voiced-highband-loss-weight 0.02 --noise-floor-loss-weight 0.03
  --waveform-discr-lrs 5e-7 5e-7 2.5e-7 --stft-discr-lr 2.5e-7
  --waveform-discr-update-every 2 2 2 --waveform-discr-loss-weights 1.0 0.25 0.25
  --stft-discr-update-every 4 --stft-discr-loss-weight 0.5
  --stage2-generator-freeze-steps 2000 --stage2-generator-hold-steps 5000 --stage2-generator-hold-lr 1e-7
  --stage2-unfreeze-encoder-rvq-step -1 --early-stopping-min-steps 150000
  --no-stage2-quality-hard-stop --gan-grad-diagnostics-every 500 --loss-grad-diagnostics-every 1000
)

# A1: RVQ is absent from both the forward signal and checkpoint selection gate.
run_stage "A1 bypass-RVQ reconstruction" 29511 "$PWD/logs/bypass-recon-$BASE.log" "$A1_APPEND_LOG" \
  --stage recon_pretrain --results-dir "$BYPASS_S1_DIR" \
  --num-train-steps "$BYPASS_STAGE1_STEPS" --bypass-rvq-during-training \
  --decoder-residual-scale-start 0.2 --decoder-residual-scale-end 1.0 \
  --decoder-residual-scale-warmup-start-steps 0 --decoder-residual-scale-warmup-end-steps 15000 \
  "${RECON_LOSSES[@]}" "${COMMON[@]}" "$A1_RESUME_FLAG"
BYPASS_S1_CKPT="$(pick_best "$BYPASS_S1_DIR")"
echo "A1 checkpoint=$BYPASS_S1_CKPT"

# A2: standard Stage-2 policy keeps the bypass Encoder fixed while Decoder and
# discriminators refine the continuous latent path with GAN losses.
run_stage "A2 bypass-RVQ GAN" 29512 "$PWD/logs/bypass-gan-$BASE.log" 0 \
  --stage gan_pretrain --results-dir "$BYPASS_S2_DIR" \
  --init-checkpoint "$BYPASS_S1_CKPT" --num-train-steps "$BYPASS_STAGE2_STEPS" \
  --bypass-rvq-during-training "${GAN_LOSSES[@]}" "${COMMON[@]}" --no-resume
BYPASS_S2_CKPT="$(pick_best "$BYPASS_S2_DIR")"
echo "A2 checkpoint=$BYPASS_S2_CKPT"

# B1: no Decoder call and no optimizer/backward step.  Only RVQ EMA and
# dead-code replacement state changes; Encoder/Decoder tensors remain bitwise fixed.
run_stage "B1 RVQ calibration with codec frozen" 29513 "$PWD/logs/rvq-calibration-$BASE.log" 0 \
  --stage recon_pretrain --results-dir "$RVQ_S1_DIR" \
  --init-checkpoint "$BYPASS_S2_CKPT" --num-train-steps "$RVQ_CALIBRATION_STEPS" \
  --rvq-calibration-only --no-bypass-rvq-during-training "${COMMON[@]}" --no-resume
RVQ_S1_CKPT="$RVQ_S1_DIR/latest.pt"
test -f "$RVQ_S1_CKPT" || { echo "ERROR: missing $RVQ_S1_CKPT" >&2; exit 2; }
echo "B1 checkpoint=$RVQ_S1_CKPT"

# B2: standard Stage-2 policy freezes Encoder and RVQ for the whole phase;
# Decoder adapts to quantization error while GAN discriminators are trained.
run_stage "B2 RVQ GAN decoder adaptation" 29514 "$PWD/logs/rvq-gan-$BASE.log" 0 \
  --stage gan_pretrain --results-dir "$RVQ_S2_DIR" \
  --init-checkpoint "$RVQ_S1_CKPT" --num-train-steps "$RVQ_STAGE2_STEPS" \
  --no-bypass-rvq-during-training "${GAN_LOSSES[@]}" "${COMMON[@]}" --no-resume

echo "===== four-phase training complete ====="
echo "bypass Stage-1: $BYPASS_S1_DIR"
echo "bypass Stage-2: $BYPASS_S2_DIR"
echo "RVQ calibration: $RVQ_S1_DIR"
echo "RVQ Stage-2: $RVQ_S2_DIR"
