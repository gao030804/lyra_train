#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/deploy/gyh/lyra_md
cd "$PROJECT"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "CONDA_PREFIX is unset. Run 'conda activate lyra' before starting this script." >&2
    exit 1
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

mkdir -p logs results

python -m py_compile \
  train_soundstream.py \
  infer_soundstream.py \
  audiolm_pytorch/data.py \
  audiolm_pytorch/soundstream.py \
  audiolm_pytorch/trainer.py

refuse_stale_fresh_run() {
    local results="$1"
    local resume="$2"

    if [[ "$resume" == "--no-resume" ]] && compgen -G "$results/*.pt" >/dev/null; then
        echo "ERROR: refusing a fresh run in $results because stale .pt files exist." >&2
        echo "Move that result directory aside, or restore latest.pt and resume intentionally." >&2
        return 1
    fi
}

run_stage0() {
    local results="results/overfit-dscnn-relu-fp-64d-16q16"
    local log="logs/overfit-dscnn-relu-fp-64d-16q16.log"
    local marker="$results/.stage_complete"

    if [[ -f "$marker" ]]; then
        echo "[stage0 overfit] already complete; skipping"
        return
    fi

    local resume="--no-resume"
    [[ -f "$results/latest.pt" ]] && resume="--resume"
    refuse_stale_fresh_run "$results" "$resume"

    echo "[stage0 overfit] starting with $resume"

    env CUDA_VISIBLE_DEVICES=0 python train_soundstream.py \
      --stage overfit \
      --audio-dir "$PROJECT/data/librispeech/LibriSpeech/train-clean-100" \
      --results-dir "$PROJECT/$results" \
      --num-train-steps 5000 \
      --batch-size 4 \
      "$resume" \
      2>&1 | tee -a "$log"

    test -f "$results/best_selected.pt"
    touch "$marker"
    echo "[stage0 overfit] complete"
}

run_6gpu_stage() {
    local stage="$1"
    local results="$2"
    local log="$3"
    local steps="$4"
    local batch="$5"
    local run_final_test="$6"
    local marker="$results/.stage_complete"

    if [[ -f "$marker" ]]; then
        echo "[$stage] already complete; skipping"
        return
    fi

    local resume="--no-resume"
    [[ -f "$results/latest.pt" ]] && resume="--resume"
    refuse_stale_fresh_run "$results" "$resume"

    local test_args=(--test-eval-batches 0)
    if [[ "$run_final_test" == "yes" ]]; then
        test_args=()
    fi

    echo "[$stage] starting with $resume"

    env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
      accelerate launch \
      --multi_gpu \
      --num_processes 6 \
      --num_machines 1 \
      --main_process_port 29501 \
      --dynamo_backend no \
      --mixed_precision no \
      train_soundstream.py \
      --stage "$stage" \
      --audio-dir "$PROJECT/data/librispeech/LibriSpeech/train-clean-100" \
      --results-dir "$PROJECT/$results" \
      --num-train-steps "$steps" \
      --batch-size "$batch" \
      --dl-num-workers 6 \
      "${test_args[@]}" \
      "$resume" \
      2>&1 | tee -a "$log"

    test -f "$results/best_selected.pt"
    touch "$marker"
    echo "[$stage] complete"
}

run_stage0

run_6gpu_stage \
  recon_pretrain \
  results/recon-pretrain-dscnn-relu-fp-64d-16q16 \
  logs/recon-pretrain-dscnn-relu-fp-64d-16q16.log \
  150000 4 no

run_6gpu_stage \
  spectral_refine \
  results/spectral-refine-dscnn-relu-fp-64d-16q16 \
  logs/spectral-refine-dscnn-relu-fp-64d-16q16.log \
  20000 4 no

run_6gpu_stage \
  gan_pretrain \
  results/gan-pretrain-dscnn-relu-fp-64d-16q16 \
  logs/gan-pretrain-dscnn-relu-fp-64d-16q16.log \
  50000 4 no

run_6gpu_stage \
  stream_finetune \
  results/stream-finetune-dscnn-relu-fp-64d-16q16 \
  logs/stream-finetune-dscnn-relu-fp-64d-16q16.log \
  20000 4 no

run_6gpu_stage \
  stream_finetune_long \
  results/stream-finetune-long-dscnn-relu-fp-64d-16q16 \
  logs/stream-finetune-long-dscnn-relu-fp-64d-16q16.log \
  20000 2 yes

echo "All non-quantized stages completed successfully at 3.2 kbps."
