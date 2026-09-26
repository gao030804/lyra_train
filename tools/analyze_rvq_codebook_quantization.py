"""Frozen-codebook PTQ, held-out audio retention, and integer export."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.integer_rvq_reference import (
    clip16, export_package, lookup, quantize_books, ratio_parameters, round_away, run_integer, requant,
)


def torch_quantized_oracle(query, books, scales, output_scale):
    """Independent PyTorch squared-distance search with integer boundaries."""
    def rq(x, ratio):
        m, s = ratio_parameters(ratio)
        product = x * m
        magnitude = product.abs()
        rounded = magnitude >> s
        if s:
            rounded += ((magnitude & (2**s-1)) >= 2**(s-1)).long()
        return torch.where(product < 0, -rounded, rounded)
    residual = torch.from_numpy(query.astype(np.int64))
    total = torch.zeros_like(residual)
    indices = []
    for q, book in enumerate(books):
        code = torch.from_numpy(book.astype(np.int64))
        index = ((residual.unsqueeze(-2)-code)**2).sum(-1).argmin(-1)
        selected = code[index]
        total += rq(selected, scales[q]/output_scale)
        residual -= selected
        if q+1 < len(books):
            residual = rq(residual, scales[q]/scales[q+1]).clamp(-32768,32767)
        indices.append(index)
    return total.clamp(-32768,32767).short().numpy(), torch.stack(indices,-1).numpy()


def nmse(a, b):
    return float(np.mean((a.astype(float)-b)**2) / max(np.mean(b.astype(float)**2), 1e-12))


def audio_metrics(target, recon, sr):
    target, recon = np.ravel(target).astype(float), np.ravel(recon).astype(float)
    n = min(len(target), len(recon))
    target, recon = target[:n], recon[:n]
    fft = 1 << (2*n-1).bit_length()
    cross = np.fft.irfft(np.fft.rfft(recon-recon.mean(), fft) *
                         np.fft.rfft(target-target.mean(), fft).conj(), fft)
    limit = min(sr//50, n-2)
    lags = np.arange(-limit, limit+1)
    lag = int(lags[np.argmax(cross[lags % fft])])
    if lag > 0:
        recon, target = recon[lag:], target[:-lag]
    elif lag < 0:
        recon, target = recon[:lag], target[-lag:]
    target, recon = target-target.mean(), recon-recon.mean()
    projection = target * (np.dot(target, recon) / max(np.dot(target, target), 1e-12))
    sdr = 10*np.log10(max(np.dot(projection, projection), 1e-12) /
                       max(np.sum((recon-projection)**2), 1e-12))
    corr = np.dot(target, recon) / max(np.linalg.norm(target)*np.linalg.norm(recon), 1e-12)
    return dict(aligned_si_sdr=float(sdr), aligned_corr=float(corr), lag=lag)


def read_manifest(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t' if path.suffix == '.tsv' else ','))
    result = []
    for row in rows:
        value = next((row.get(k) for k in ('source', 'dataset_source', 'source_file', 'original_copy', 'original') if row.get(k)), None)
        if not value:
            raise ValueError(f'Manifest has no source column: {path}')
        p = Path(value)
        if not p.is_absolute():
            p = path.parent / p
        result.append(p.resolve(strict=True))
    if not result or len(set(result)) != len(result):
        raise ValueError('Manifest must be nonempty with unique paths')
    return result


def extract_books(model):
    from torch import nn
    if model.rq_groups != 1 or model.rq_use_cosine_sim or model.use_lookup_free_quantizer or model.use_finite_scalar_quantizer:
        raise ValueError('Only single-group Euclidean RVQ is supported')
    rvq = model.rq.rvqs[0]
    for owner in (rvq, *rvq.layers):
        for name in ('project_in', 'project_out'):
            if not isinstance(getattr(owner, name, nn.Identity()), nn.Identity):
                raise ValueError('Internal VQ projections unsupported; external codec projections are preserved')
    books = []
    for layer in rvq.layers:
        cb = layer._codebook
        if hasattr(cb, 'initted') and not bool(cb.initted.all()):
            raise ValueError('Codebook is uninitialized')
        e = cb.embed.detach().cpu().numpy()
        if e.ndim != 3 or e.shape[0] != 1:
            raise ValueError(f'Expected [1,K,D] codebook, got {e.shape}')
        books.append(e[0].astype(np.float64))
    return books


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--calibration-manifest', type=Path, required=True)
    p.add_argument('--evaluation-manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--segment-seconds', type=float, default=4.)
    p.add_argument('--percentiles', nargs='+', type=float, default=[100., 99.99, 99.9, 99.5])
    p.add_argument('--max-si-sdr-drop', type=float, default=.05)
    p.add_argument('--max-p10-si-sdr-drop', type=float, default=.20)
    p.add_argument('--max-worst-si-sdr-drop', type=float, default=.30)
    p.add_argument('--max-corr-drop', type=float, default=.01)
    p.add_argument('--max-vqout-nmse', type=float, default=.04)
    p.add_argument('--export', action='store_true', help='Export only if held-out retention gates pass')
    args = p.parse_args()
    from infer_soundstream import load_audio, load_checkpoint, soundstream_from_checkpoint
    if args.segment_seconds <= 0 or any(x <= 0 or x > 100 for x in args.percentiles):
        raise ValueError('Invalid duration/percentiles')
    if min(args.max_si_sdr_drop, args.max_corr_drop, args.max_vqout_nmse) < 0:
        raise ValueError('Retention tolerances must be nonnegative')
    cal_paths = read_manifest(args.calibration_manifest)
    eval_paths = read_manifest(args.evaluation_manifest)
    if set(cal_paths) & set(eval_paths):
        raise ValueError('Calibration and evaluation audio must be disjoint')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    pkg = load_checkpoint(args.checkpoint)
    model = soundstream_from_checkpoint(pkg, use_ema=False).to(args.device).eval()
    model.requires_grad_(False)
    if getattr(model, 'hardware_encoder_qat', False):
        accepted = (pkg.get('hardware_qat_progression_state') or {}).get('accepted_groups')
        full_qat_proven = (pkg.get('weight_source') == 'online_full_int8_qat_gated' or
                          (isinstance(accepted, (list, tuple)) and len(accepted) == 6 and all(accepted)))
        if not full_qat_proven:
            raise ValueError('Use a validated full Encoder QAT checkpoint')
        model.set_hardware_qat_sensitivity_group(-1, accepted_groups=range(6))
        if not model.hardware_qat_is_full():
            raise ValueError('Encoder full-QAT restoration failed')
    books = extract_books(model)
    original_books = [b.copy() for b in books]
    print(f'Frozen topology: {len(books)} stages, K={[len(b) for b in books]}, D={books[0].shape[1]}', flush=True)
    sr = int(model.target_sample_hz)
    length = int(args.segment_seconds * sr)//model.downsample_factor*model.downsample_factor
    if length < model.downsample_factor:
        raise ValueError('Segment shorter than one codec frame')

    def collect(paths):
        records = []
        for path in paths:
            wave = load_audio(path, sr)
            if wave.shape[-1] < length:
                raise ValueError(f'Audio too short: {path}')
            start = (wave.shape[-1]-length)//2
            wave = wave[:, start:start+length].unsqueeze(0).to(args.device)
            encoded = model.encoder(wave).transpose(1, 2)
            if model.encoder_attn is not None:
                encoded = model.encoder_attn(encoded)
            query = model.rq_input_projection(encoded)
            quantized, indices, _ = model.rq(query, freeze_codebook=True)
            selected_sum = sum(book[indices[0,...,q].cpu().numpy()] for q,book in enumerate(books))
            np.testing.assert_allclose(quantized.cpu().numpy(), selected_sum, rtol=1e-4, atol=1e-5,
                                       err_msg='RVQ includes an unsupported transform or codebook lookup rule')
            baseline = model.decode(model.rq_output_projection(quantized))
            records.append(dict(path=str(path), start_sample=start, wave=wave.cpu().numpy(),
                                query=query.cpu().numpy(), quantized=quantized.cpu().numpy(),
                                indices=indices[0].cpu().numpy(), baseline=baseline.cpu().numpy()))
        return records

    calibration = collect(cal_paths)
    evaluation = collect(eval_paths)
    scales = [max(np.max(np.abs(b))/127., 1e-12) for b in books]
    # Fixed output scale selected once from calibration FP RVQ output only.
    output_scale = max(max(np.max(np.abs(r['quantized'])) for r in calibration)/32767., 1e-12)

    def assess(scales, records, save_audio=False, verify=False):
        qbooks = quantize_books(books, scales)
        rows, traces = [], []
        for i, record in enumerate(records):
            raw = round_away(record['query']/scales[0])
            query = clip16(raw)
            value, indices, trace = run_integer(query, qbooks, scales, output_scale)
            if verify:
                oracle, oracle_indices, _ = run_integer(query, qbooks, scales, output_scale, squared_distance=True)
                np.testing.assert_array_equal(indices, oracle_indices)
                np.testing.assert_array_equal(value, oracle)
                np.testing.assert_array_equal(value, lookup(indices, qbooks, scales, output_scale))
                torch_value, torch_indices = torch_quantized_oracle(query, qbooks, scales, output_scale)
                np.testing.assert_array_equal(indices, torch_indices)
                np.testing.assert_array_equal(value, torch_value)
            latent = torch.from_numpy(value.astype(np.float32)*output_scale).to(args.device)
            audio = model.decode(model.rq_output_projection(latent)).cpu().numpy()
            before = audio_metrics(record['wave'], record['baseline'], sr)
            after = audio_metrics(record['wave'], audio, sr)
            pre_output=sum(requant(book[indices[...,q]].astype(np.int64),*ratio_parameters(scales[q]/output_scale)) for q,book in enumerate(qbooks))
            row = dict(path=record['path'], start_sample=record['start_sample'],
                       aligned_si_sdr_delta=after['aligned_si_sdr']-before['aligned_si_sdr'],
                       aligned_corr_delta=after['aligned_corr']-before['aligned_corr'],
                       vqout_nmse=nmse(value.astype(float)*output_scale, record['quantized']),
                       input_saturation=int(np.count_nonzero((raw < -32768)|(raw > 32767))),
                       residual_saturation=sum(t['saturation'] for t in trace),
                       output_saturation=int(np.count_nonzero((pre_output < -32768)|(pre_output > 32767))))
            teacher_residual=record['query'].astype(float).copy()
            base_energy=max(float(np.mean(teacher_residual**2)),1e-12)
            for q in range(len(books)):
                row[f'q{q:02d}_index_flip'] = float(np.mean(indices[...,q] != record['indices'][...,q]))
                row[f'q{q:02d}_selected_codeword_nmse'] = nmse(qbooks[q][indices[...,q]].astype(float)*scales[q], books[q][record['indices'][...,q]])
                row[f'q{q:02d}_residual_rms'] = float(np.sqrt(np.mean((trace[q]['residual']*scales[q])**2)))
                selected=books[q][record['indices'][...,q]]
                energy=np.mean(teacher_residual**2,axis=-1)
                flip=indices[...,q] != record['indices'][...,q]
                row[f'q{q:02d}_weighted_index_flip']=float(np.mean(flip*energy)/base_energy)
                row[f'q{q:02d}_residual_energy']=float(energy.mean())
                row[f'q{q:02d}_selected_codeword_energy']=float(np.mean(selected**2))
                teacher_residual-=selected
            row['final_residual_rms'] = float(np.sqrt(np.mean((trace[-1]['after_subtract']*scales[-1])**2)))
            rows.append(row)
            traces.append((query, value, indices, trace))
            if save_audio:
                import soundfile as sf
                for label, data in [('original', record['wave']), ('fp_codebook', record['baseline']), ('int8_codebook', audio)]:
                    sf.write(str(out/f'{i:02d}_{label}.wav'), data.reshape(-1), sr, subtype='FLOAT')
        means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('path', 'start_sample')}
        means['p10_si_sdr_delta']=float(np.percentile([r['aligned_si_sdr_delta'] for r in rows],10))
        means['worst_si_sdr_delta']=min(r['aligned_si_sdr_delta'] for r in rows)
        means['saturation_count']=sum(r['input_saturation']+r['residual_saturation']+r['output_saturation'] for r in rows)
        return means, rows, traces

    def eligible(metrics):
        return (metrics['aligned_si_sdr_delta'] >= -args.max_si_sdr_drop and
                metrics['p10_si_sdr_delta'] >= -args.max_p10_si_sdr_drop and
                metrics['worst_si_sdr_delta'] >= -args.max_worst_si_sdr_drop and
                metrics['aligned_corr_delta'] >= -args.max_corr_drop and
                metrics['vqout_nmse'] <= args.max_vqout_nmse and metrics['saturation_count']==0)

    def rank(metrics, element_nmse):
        early = sum(metrics[f'q{q:02d}_weighted_index_flip'] for q in range(min(3, len(books))))
        return (not eligible(metrics), -metrics['aligned_si_sdr_delta'], -metrics['p10_si_sdr_delta'],
                -metrics['worst_si_sdr_delta'], -metrics['aligned_corr_delta'], metrics['vqout_nmse'], early, element_nmse)

    max_scale_metrics,_,_=assess(scales,calibration,verify=True)
    max_scales=scales.copy()
    trials = []
    for q, book in enumerate(books):
        choices = []
        # Include current value; coordinate search evaluates the complete RVQ/audio chain.
        for percentile in dict.fromkeys([100., *args.percentiles]):
            trial = scales.copy()
            trial[q] = max(float(np.percentile(np.abs(book), percentile))/127., 1e-12)
            metrics, _, _ = assess(trial, calibration)
            cb = quantize_books([book], [trial[q]])[0]
            element_nmse = nmse(cb.astype(float)*trial[q], book)
            trials.append(dict(stage=q, percentile=percentile, scale=trial[q], codebook_nmse=element_nmse, metrics=metrics))
            choices.append((rank(metrics, element_nmse), trial[q]))
        scales[q] = min(choices, key=lambda x:x[0])[1]
        print(f'stage {q}: selected scale={scales[q]:.9g}', flush=True)
    metrics, rows, traces = assess(scales, evaluation, save_audio=True, verify=True)
    passed = eligible(metrics)
    for a, b in zip(original_books, extract_books(model)):
        np.testing.assert_array_equal(a, b)  # EMA must never mutate master codebooks
    qbooks = quantize_books(books, scales)
    report = dict(topology=dict(sizes=[len(b) for b in books], dimension=books[0].shape[1]),
                  checkpoint=str(args.checkpoint.resolve()), weights='online',
                  calibration_manifest=str(args.calibration_manifest.resolve()),
                  evaluation_manifest=str(args.evaluation_manifest.resolve()),
                  calibration_sources=[str(path) for path in cal_paths],
                  scopes='QAT Encoder fixed; external projections and audio Decoder remain float',
                  scales=scales, output_scale=output_scale, metrics=metrics, per_file=rows,
                  codebook_nmse=[nmse(b.astype(float)*s, fp) for b,s,fp in zip(qbooks,scales,books)],
                  calibration_trials=trials, retention_pass=bool(passed),scale_mode='v1',
                  max_scale_baseline=dict(scales=max_scales,calibration_metrics=max_scale_metrics),
                  thresholds=dict(si_sdr_drop=args.max_si_sdr_drop,p10_si_sdr_drop=args.max_p10_si_sdr_drop,
                                  worst_si_sdr_drop=args.max_worst_si_sdr_drop,saturation_count=0,
                                  corr_drop=args.max_corr_drop, vqout_nmse=args.max_vqout_nmse),
                  integer_squared_distance_oracle_pass=True, decoder_lookup_exact=True,
                  pytorch_quantized_oracle_pass=True,
                  rtl_verified=False,rtl_index_mismatch_count=None,rtl_latent_mismatch_count=None,
                  codebook_storage_bytes=dict(int8=sum(b.size for b in books),fp32=4*sum(b.size for b in books)))
    (out/'rvq_ptq_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.export and passed:
        digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        export_package(out/'export', qbooks, scales, output_scale,
                       provenance=dict(checkpoint=str(args.checkpoint.resolve()), sha256=digest, weights='online'))
        # Full per-stage scores/residuals, for every validation latent frame.
        for i, (query, value, indices, trace) in enumerate(traces):
            files = dict(input_int16=query, output_int16=value, indices=indices.astype('<i4'))
            for q, t in enumerate(trace):
                files.update({f'q{q:02d}_residual': t['residual'].astype('<i2'),
                              f'q{q:02d}_scores': t['scores'].astype('<i4'),
                              f'q{q:02d}_after_subtract': t['after_subtract'].astype('<i4')})
            np.savez(out/f'golden_{i:02d}.npz', **files)
        print('Export and per-stage goldens complete.', flush=True)
    print(f'Retention pass={passed}; report={out / "rvq_ptq_report.json"}', flush=True)
    if args.export and not passed:
        raise SystemExit('Held-out retention failed; no deployment package exported.')


if __name__ == '__main__':
    main()
