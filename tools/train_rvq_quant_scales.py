"""Standalone bounded RVQ scale training with integer validation selection."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import random
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.rvq_scale_qat import ScaleRVQ, integer_encoder, make_projection, project_integer, export_projection, retention_summary
from tools.analyze_rvq_codebook_quantization import read_manifest, extract_books, audio_metrics, nmse
from tools.integer_rvq_reference import clip16, round_away, quantize_books, run_integer, export_package, lookup


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','ptq-report','train-manifest','validation-manifest','test-manifest','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--max-ratio',type=float,default=1.15)
    p.add_argument('--topk',type=int,default=8)
    p.add_argument('--temperature',type=float,default=.1)
    p.add_argument('--segment-seconds',type=float,default=4.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args()
    if min(args.steps,args.eval_every,args.lr,args.segment_seconds)<=0:
        raise ValueError('Steps, LR, interval and duration must be positive')
    from infer_soundstream import load_checkpoint,soundstream_from_checkpoint,load_audio
    from tools.export_encoder_integer_goldens import full_qat_proven
    torch.manual_seed(args.seed)
    rng=random.Random(args.seed)
    paths=[read_manifest(x) for x in (args.train_manifest,args.validation_manifest,args.test_manifest)]
    if any(set(paths[i]) & set(paths[j]) for i in range(3) for j in range(i)):
        raise ValueError('Training/validation/test manifests must be disjoint')
    # Reject renamed duplicate audio too; test audio must not optimize scales.
    hashes=[{digest(x) for x in split} for split in paths]
    if any(hashes[i]&hashes[j] for i in range(3) for j in range(i)):
        raise ValueError('Audio contents overlap across dataset splits')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    pkg=load_checkpoint(args.checkpoint)
    if not full_qat_proven(pkg):
        raise ValueError('A validated full Encoder-QAT checkpoint is required')
    model=soundstream_from_checkpoint(pkg,use_ema=False).to(args.device).eval()
    model.requires_grad_(False)
    model.set_hardware_qat_sensitivity_group(-1,accepted_groups=range(6))
    if not model.hardware_qat_is_full():
        raise ValueError('Failed to restore full Encoder QAT state')
    if model.hardware_qat_activation_bits!=16 or model.hardware_qat_accumulator_bits!=40:
        raise ValueError('W8A16/ACC40 Encoder required')
    if model.encoder_attn is not None or not isinstance(model.rq_input_projection,torch.nn.Linear):
        raise ValueError('Requires plain Encoder followed by one Linear input projection')
    books=extract_books(model)
    ptq=json.loads(args.ptq_report.read_text(encoding='utf-8'))
    if ptq['topology'] != dict(sizes=[len(b) for b in books],dimension=books[0].shape[1]):
        raise ValueError('PTQ topology differs from checkpoint')
    initial=np.asarray(ptq['scales'],dtype=float)
    output_scale=float(ptq['output_scale'])
    sr=int(model.target_sample_hz)
    length=int(args.segment_seconds*sr)//model.downsample_factor*model.downsample_factor
    if length<model.downsample_factor:
        raise ValueError('Segment too short')
    projection=None
    @torch.no_grad()
    def collect(split):
        nonlocal projection
        records=[]
        for i,path in enumerate(split):
            wave=load_audio(path,sr)
            if wave.shape[-1]<length:
                raise ValueError(f'Audio too short: {path}')
            start=(wave.shape[-1]-length)//2
            wave=wave[:,start:start+length].unsqueeze(0)
            # The integer Encoder consumes the same explicitly rounded PCM as teacher.
            pcm=clip16(round_away(wave.numpy()*32768))
            wave=torch.from_numpy(pcm.astype(np.float32)/32768)
            encoder_int,encoder_scale=integer_encoder(model,wave)
            if projection is None:
                linear=model.rq_input_projection
                projection=make_projection(linear.weight.detach().cpu().numpy(),
                    None if linear.bias is None else linear.bias.detach().cpu().numpy(),encoder_scale,initial[0])
            if not np.isclose(encoder_scale,projection['input_scale'],rtol=1e-6,atol=0):
                raise RuntimeError('Encoder output scale changed')
            query_int,saturation=project_integer(encoder_int.transpose(1,2).numpy(),projection)
            fp_query=model.rq_input_projection(model.encoder(wave.to(args.device)).transpose(1,2))
            fp_out,fp_indices,_=model.rq(fp_query,freeze_codebook=True)
            selected_sum=sum(book[fp_indices[0,...,q].cpu().numpy()] for q,book in enumerate(books))
            np.testing.assert_allclose(fp_out.cpu().numpy(),selected_sum,rtol=1e-4,atol=1e-5)
            audio=model.decode(model.rq_output_projection(fp_out)).cpu().numpy()
            records.append(dict(path=str(path),start_sample=start,pcm=pcm,encoder_int=encoder_int.transpose(1,2).numpy(),
                query_int=query_int,query=query_int.astype(float)*initial[0],teacher_query=fp_query.cpu().numpy(),
                teacher_out=fp_out.cpu().numpy(),teacher_indices=fp_indices[0].cpu().numpy(),
                teacher_audio=audio,wave=wave.numpy(),projection_saturation=saturation))
            print(f'cached {i+1}/{len(split)} {path.name}',flush=True)
        return records
    train=collect(paths[0])
    validation=collect(paths[1])
    student=ScaleRVQ(books,initial,output_scale,args.max_ratio,args.topk,args.temperature).to(args.device)
    optimizer=torch.optim.Adam(student.parameters(),lr=args.lr,weight_decay=0)
    provenance=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=digest(args.checkpoint),
                    ptq_report=str(args.ptq_report.resolve()),ptq_report_sha256=digest(args.ptq_report),
                    train_sources=[str(x) for x in paths[0]],validation_sources=[str(x) for x in paths[1]],
                    test_sources=[str(x) for x in paths[2]],args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (args.output_dir/'run_config.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')

    @torch.no_grad()
    def evaluate(records,step,save=False):
        scales=student.scales().cpu().numpy()
        qbooks=quantize_books(books,scales)
        rows=[]
        for i,r in enumerate(records):
            y,indices,trace=run_integer(r['query_int'],qbooks,scales,output_scale,residual_scales=initial)
            oracle,oi,_=run_integer(r['query_int'],qbooks,scales,output_scale,residual_scales=initial,squared_distance=True)
            np.testing.assert_array_equal(indices,oi)
            np.testing.assert_array_equal(y,oracle)
            np.testing.assert_array_equal(y,lookup(indices,qbooks,scales,output_scale))
            # Count final output clipping as well as search/state boundaries.
            from tools.integer_rvq_reference import requant,ratio_parameters
            pre_output=sum(requant(book[indices[...,q]].astype(np.int64),*ratio_parameters(scales[q]/output_scale)) for q,book in enumerate(qbooks))
            sat=r['projection_saturation']+sum(t['saturation']+t['search_saturation'] for t in trace)+int(np.count_nonzero((pre_output < -32768)|(pre_output > 32767)))
            decoded=model.decode(model.rq_output_projection(torch.tensor(y.astype(float)*output_scale,device=args.device,dtype=torch.float32))).cpu().numpy()
            before=audio_metrics(r['wave'],r['teacher_audio'],sr)
            after=audio_metrics(r['wave'],decoded,sr)
            row=dict(path=r['path'],start_sample=r['start_sample'],aligned_si_sdr_delta=after['aligned_si_sdr']-before['aligned_si_sdr'],
                     aligned_corr_delta=after['aligned_corr']-before['aligned_corr'],vqout_nmse=nmse(y.astype(float)*output_scale,r['teacher_out']),saturation=sat)
            residual=r['teacher_query'].astype(float).copy()
            base_energy=max(float(np.mean(residual**2)),1e-12)
            for q,book in enumerate(books):
                distances=((residual[...,None,:]-book)**2).sum(-1)
                margin=np.sort(distances,axis=-1)[...,1]-distances.min(-1)
                qmargin=(np.sort(trace[q]['scores'].astype(float),axis=-1)[...,1]-trace[q]['scores'].min(-1))*scales[q]**2
                flip=indices[...,q]!=r['teacher_indices'][...,q]
                row[f'q{q:02d}']=dict(index_flip=float(flip.mean()),energy_weighted_flip=float(flip.mean()*np.mean(residual**2)/base_energy),
                    teacher_margin=float(margin.mean()),integer_margin=float(qmargin.mean()),
                    teacher_margin_at_flip=float(margin[flip].mean()) if flip.any() else None,
                    integer_margin_at_flip=float(qmargin[flip].mean()) if flip.any() else None)
                residual-=book[r['teacher_indices'][...,q]]
            rows.append(row)
            if save:
                import soundfile as sf
                for label,audio in [('original',r['wave']),('fp_codebook',r['teacher_audio']),('scale_qat',decoded)]:
                    sf.write(str(args.output_dir/f'{i:02d}_{label}.wav'),audio.reshape(-1),sr,subtype='FLOAT')
                golden=dict(pcm=r['pcm'],encoder_int16=r['encoder_int'],projection_int16=r['query_int'],indices=indices,output_int16=y)
                for q,t in enumerate(trace):
                    for key in ('stored_residual','residual','scores','after_subtract'):
                        golden[f'q{q:02d}_{key}']=t[key]
                np.savez(args.output_dir/f'golden_{i:02d}.npz',**golden)
        report=dict(step=step,scales=scales.tolist(),summary=retention_summary(rows),per_file=rows,rtl_verified=False)
        return report

    best_key=None
    best_step=None
    def validate(step):
        nonlocal best_key,best_step
        report=evaluate(validation,step)
        (args.output_dir/f'validation_{step:05d}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        s=report['summary']
        print(f'integer validation step={step} {s}',flush=True)
        key=(s['mean_vqout_nmse'],-s['worst_si_sdr_delta'],-s['p10_si_sdr_delta'])
        if s['passed'] and (best_key is None or key<best_key):
            best_key,best_step=key,step
            torch.save(dict(scale_state=student.state_dict(),step=step,validation=report,provenance=provenance),args.output_dir/'best_scales.pt')
    validate(0)
    for step in range(1,args.steps+1):
        r=rng.choice(train)
        query=torch.tensor(r['query'],device=args.device)
        teacher_query=torch.tensor(r['teacher_query'],device=args.device)
        target=torch.tensor(r['teacher_out'],device=args.device)
        y,dist,cb=student(query,teacher_query)
        vq=(y-target).square().mean()/target.square().mean().clamp_min(1e-12)
        loss=vq+.2*dist+.02*cb
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if student.delta.grad is None or not torch.isfinite(student.delta.grad).all():
            raise RuntimeError('Scale gradient missing or nonfinite')
        grad=float(student.delta.grad.norm())
        torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
        optimizer.step()
        if step%25==0:
            print(f'{step}: loss={loss.item():.6f} vqout={vq.item():.6f} distance={dist.item():.6f} codebook={cb.item():.6f} scale_grad={grad:.6g}',flush=True)
        if step%args.eval_every==0 or step==args.steps:
            validate(step)
            torch.save(dict(scale_state=student.state_dict(),optimizer=optimizer.state_dict(),step=step,provenance=provenance),args.output_dir/'latest_scales.pt')
    for before,after in zip(books,extract_books(model)):
        np.testing.assert_array_equal(before,after)
    if best_step is None:
        raise SystemExit('No integer-validation eligible scale checkpoint; see validation reports, no export.')
    saved=torch.load(args.output_dir/'best_scales.pt',map_location=args.device,weights_only=False)
    student.load_state_dict(saved['scale_state'])
    test=collect(paths[2])  # test is not consulted for checkpoint selection
    final=evaluate(test,best_step,save=True)
    (args.output_dir/'final_test_report.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    if not final['summary']['passed']:
        raise SystemExit('Final test retention failed; no deployment export.')
    scales=student.scales().detach().cpu().numpy()
    manifest=export_package(args.output_dir/'export',quantize_books(books,scales),scales,output_scale,provenance=provenance,residual_scales=initial)
    export_projection(args.output_dir/'export',projection)
    manifest.update(projection_quantized=True,projection_manifest='projection_manifest.json',
                    output_projection_quantized=False,rtl_verified=False)
    (args.output_dir/'export'/'rvq_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Complete: best step={best_step}, export={args.output_dir / "export"}',flush=True)


if __name__=='__main__':
    main()
