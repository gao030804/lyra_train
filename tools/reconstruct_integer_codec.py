"""Diagnostic binary-parameter codec reconstruction, offline and stateful.

Encoder graph/float Decoder come from a trusted checkpoint. Encoder/RVQ
arithmetic reads exported binaries. Input projection is exported and read back.
This tool does not train, select candidates or approve deployment.
"""
import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.integer_codec_package import BinaryEncoder, BinaryRVQ, digest, unpack_weight
from tools.integer_rvq_reference import clip16, round_away
from tools.rvq_scale_qat import make_projection, export_projection, project_integer


def encoder_graph(module,name=''):
    from audiolm_pytorch.soundstream import CausalConv1d, Residual, DepthwiseSeparableCausalConv1d, LowRankPointwiseCausalConv1d
    from audiolm_pytorch.hardware_quantization import HardwareReLU
    def path(k):
        return f'{name}.{k}' if name else k
    if isinstance(module,CausalConv1d):
        if module.pad_mode != 'constant':
            raise ValueError('Integer reconstruction requires zero causal padding')
        return dict(kind='conv',name=name)
    if isinstance(module,Residual):
        return dict(kind='residual',name=name,branch=encoder_graph(module.fn,path('fn')))
    if isinstance(module,(DepthwiseSeparableCausalConv1d,LowRankPointwiseCausalConv1d)):
        return encoder_graph(module.net,path('net'))
    if isinstance(module,torch.nn.Sequential):
        return dict(kind='seq',children=[encoder_graph(v,path(k)) for k,v in module.named_children()])
    if isinstance(module,HardwareReLU):
        return dict(kind='relu',name=name,scale=float(module.fake_quant.scale.item()))
    if isinstance(module,torch.nn.Identity):
        return dict(kind='identity')
    raise TypeError(f'Unsupported Encoder graph node: {name}: {type(module).__name__}')


def load_projection(root):
    root = Path(root)
    m = json.loads((root/'projection_manifest.json').read_text())
    spec = dict(m)
    spec['weight'] = unpack_weight((root/'projection_weights_4x8_int8.bin').read_bytes(),m['cout'],m['cin'],1)[:,:,0].astype(np.int8)
    spec['bias'] = np.fromfile(root/'projection_bias_int40_in_int64.bin',dtype='<i8')
    spec['multiplier'] = np.fromfile(root/'projection_multiplier_int32.bin',dtype='<i4')
    spec['shift'] = np.fromfile(root/'projection_shift_uint8.bin',dtype='u1')
    return spec


def check_stream_decoder(module):
    """Do not let stream_module's stateless fallback hide unsupported state."""
    from audiolm_pytorch.soundstream import CausalConv1d, CausalConvTranspose1d, CausalLinearUpsampleConv1d, DepthwiseSeparableCausalConv1d, Residual
    if isinstance(module,(CausalConv1d,CausalConvTranspose1d,CausalLinearUpsampleConv1d)):
        return
    if isinstance(module,DepthwiseSeparableCausalConv1d):
        check_stream_decoder(module.net)
        return
    if isinstance(module,Residual):
        if getattr(module,'hardware_residual_add',None) is not None:
            raise ValueError('Decoder must remain floating; quantized residual Decoder unsupported')
        check_stream_decoder(module.fn)
        return
    if isinstance(module,torch.nn.Sequential):
        for child in module:
            check_stream_decoder(child)
        return
    if isinstance(module,(torch.nn.ReLU,torch.nn.ELU,torch.nn.Identity,torch.nn.Tanh)):
        return
    raise TypeError(f'Unsupported stateful Decoder node: {type(module).__name__}')


def waveform_metrics(target,recon):
    t = np.asarray(target,dtype=np.float64).reshape(-1)
    r = np.asarray(recon,dtype=np.float64).reshape(-1)
    if t.shape != r.shape or not np.isfinite(r).all():
        raise ValueError('Invalid reconstruction shape/values')
    tc,rc = t-t.mean(),r-r.mean()
    alpha = np.dot(tc,rc)/max(np.dot(tc,tc),1e-12)
    return dict(si_sdr=10*math.log10(max(np.sum((alpha*tc)**2),1e-12)/max(np.sum((rc-alpha*tc)**2),1e-12)),
                correlation=float(np.dot(tc,rc)/max(np.linalg.norm(tc)*np.linalg.norm(rc),1e-12)),
                mse=float(np.mean((t-r)**2)),mae=float(np.mean(np.abs(t-r))),
                peak=float(np.max(np.abs(r))),clip_fraction=float(np.mean(np.abs(r)>=1)))


def jobs(args):
    if args.audio:
        return [dict(path=str(args.audio),start_sample=args.start_sample)]
    if args.report:
        rows = json.loads(args.report.read_text(encoding='utf-8'))['per_file']
    else:
        with args.manifest.open(encoding='utf-8-sig',newline='') as f:
            rows = list(csv.DictReader(f))
        rows = [dict(path=r['source'],start_sample=int(r.get('start_sample') or 0)) for r in rows]
    return rows[args.file_index:args.file_index+args.num_files]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('encoder-dir','rvq-dir','checkpoint','output-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--audio',type=Path)
    source.add_argument('--report',type=Path,help='Reuse per_file path/start_sample from calibrated test JSON')
    source.add_argument('--manifest',type=Path,help='CSV source and optional start_sample')
    p.add_argument('--mode',choices=['offline','compare'],default='offline',help='compare additionally processes 320-sample frames and asserts consistency')
    p.add_argument('--start-sample',type=int,default=0)
    p.add_argument('--segment-seconds',type=float,default=4,help='0 = full file from sample zero, ignoring crop start')
    p.add_argument('--num-files',type=int,default=10)
    p.add_argument('--file-index',type=int,default=0)
    p.add_argument('--device',default='cuda')
    p.add_argument('--decoder-atol',type=float,default=1e-5)
    p.add_argument('--decoder-rtol',type=float,default=1e-4)
    args = p.parse_args()
    if min(args.start_sample,args.segment_seconds,args.file_index,args.decoder_atol,args.decoder_rtol)<0 or args.num_files<1:
        p.error('Invalid range/duration/tolerance')
    from infer_soundstream import load_checkpoint, soundstream_from_checkpoint
    from audiolm_pytorch.hardware_export import export_hardware_encoder_package
    from audiolm_pytorch.soundstream import stream_module
    from tools.export_encoder_integer_goldens import full_qat_proven
    from tools.analyze_rvq_codebook_quantization import audio_metrics, nmse
    import soundfile as sf

    selected = jobs(args)
    if not selected:
        raise ValueError('Empty audio selection')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    model_pkg = load_checkpoint(args.checkpoint)
    if not full_qat_proven(model_pkg):
        raise ValueError('Checkpoint must prove full Encoder QAT')
    model = soundstream_from_checkpoint(model_pkg,use_ema=False).cpu().eval()
    model.requires_grad_(False)
    model.set_hardware_qat_sensitivity_group(-1,accepted_groups=range(6))
    if model.encoder_attn is not None or model.decoder_attn is not None:
        raise ValueError('Attention not supported by this streaming verifier')
    if args.mode == 'compare':
        check_stream_decoder(model.decoder)
    if model.target_sample_hz != 16000 or model.downsample_factor != 320:
        raise ValueError('Expected 16 kHz and downsample factor 320')
    graph = encoder_graph(model.encoder)
    enc = BinaryEncoder(args.encoder_dir,graph)
    rvq = BinaryRVQ(args.rvq_dir)
    sha = digest(args.checkpoint)
    if enc.manifest.get('source_checkpoint_sha256') != sha:
        raise ValueError('Encoder export/checkpoint SHA mismatch')
    rvq_sha = rvq.manifest.get('provenance',{}).get('training_provenance',{}).get('checkpoint_sha256')
    if rvq_sha != sha:
        raise ValueError('RVQ candidate/checkpoint SHA mismatch; expected calibrated diagnostic export')
    # Regeneration is only an integrity check. Actual execution still reads bins.
    with tempfile.TemporaryDirectory(prefix='encoder-package-check-') as tmp:
        regenerated = json.loads(export_hardware_encoder_package(model,tmp).read_text())
        for path in Path(tmp).glob('*.bin'):
            if not (args.encoder_dir/path.name).is_file() or digest(path)!=digest(args.encoder_dir/path.name):
                raise ValueError(f'Encoder blob differs from checkpoint: {path.name}')
        for field in ('layers','residuals'):
            if regenerated[field] != enc.manifest[field]:
                raise ValueError(f'Encoder {field} metadata mismatch; re-export matching checkpoint')
    # Missing projection package is generated once, frozen, and read back.
    linear = model.rq_input_projection
    if not isinstance(linear,torch.nn.Linear):
        raise ValueError('Expected Linear 64->32 input projection')
    proj = make_projection(linear.weight.numpy(),None if linear.bias is None else linear.bias.numpy(),
                           enc.manifest['layers'][-1]['output_scale'],rvq.manifest['input_scale'])
    projdir = args.output_dir/'projection_export'
    export_projection(projdir,proj)
    loaded_proj = load_projection(projdir)
    for field in ('weight','bias','multiplier','shift'):
        np.testing.assert_array_equal(proj[field],loaded_proj[field])
    proj = loaded_proj
    model = model.to(args.device)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    def latent_tensor(y):
        return torch.from_numpy(y.astype(np.float32)*rvq.manifest['output_scale']).unsqueeze(0).to(args.device)
    rows = []
    report = dict(mode=args.mode,diagnostic_only=True,deployment_approved=False,rtl_verified=False,software_checks_passed=False,
                  checkpoint_sha256=sha,encoder_graph=graph,per_file=rows,
                  parameter_sha256={str(f.resolve()):digest(f) for d in (args.encoder_dir,args.rvq_dir,projdir) for f in d.iterdir() if f.suffix in ('.bin','.json')},
                  decoder_atol=args.decoder_atol,decoder_rtol=args.decoder_rtol,
                  reference='full QAT Encoder + floating projection and codebook + floating Decoder')
    for i,job in enumerate(selected):
        print(f'[{i+1}/{len(selected)}] loading {job["path"]}',flush=True)
        audio,sr = sf.read(job['path'],dtype='float32',always_2d=True)
        if sr != 16000:
            raise ValueError('Input must be 16 kHz (no implicit resampling)')
        start = 0 if args.segment_seconds == 0 else int(job.get('start_sample',0))
        length = len(audio)-start if args.segment_seconds == 0 else round(args.segment_seconds*sr)
        if start<0 or length<320 or start+length>len(audio):
            raise ValueError(f'Invalid segment: {job["path"]}')
        wave = audio[start:start+length].mean(axis=1)
        if not np.isfinite(wave).all():
            raise ValueError('Input audio contains non-finite values')
        pcm = clip16(round_away(wave*32768))
        pad = (-len(pcm))%320
        pcm = np.pad(pcm,(0,pad))
        enc.reset()
        encoded,scale,trace = enc.run(pcm)
        if not math.isclose(scale,proj['input_scale'],rel_tol=1e-6):
            raise ValueError('Encoder output/projection input scale mismatch')
        if encoded.shape[-1] != len(pcm)//320:
            raise ValueError('Encoder output frame count mismatch')
        query,proj_sat = project_integer(encoded.T,proj)
        y,index,rvq_sat = rvq.encode(query)
        # Decode saved indices, not the Encoder's floating latent.
        destination = args.output_dir/f'{i:02d}'
        destination.mkdir()
        np.save(destination/'indices.npy',index)
        restored,_ = rvq.decode(np.load(destination/'indices.npy',allow_pickle=False))
        np.testing.assert_array_equal(y,restored)
        decoded = model.decode(model.rq_output_projection(latent_tensor(restored))).cpu().numpy().reshape(-1)
        teacher_wave = torch.from_numpy(pcm.astype(np.float32)/32768).reshape(1,1,-1).to(args.device)
        reference_query = model.rq_input_projection(model.encoder(teacher_wave).transpose(1,2))
        reference_y,_,_ = model.rq(reference_query,freeze_codebook=True)
        reference = model.decode(model.rq_output_projection(reference_y)).cpu().numpy().reshape(-1)
        if min(len(decoded),len(reference)) != len(pcm) or len(decoded)!=len(reference):
            raise ValueError('Unexpected Decoder length; no silent temporal trimming')
        row = dict(path=job['path'],start_sample=start,num_samples=length,padding_samples=pad,
                   encoder_saturation=dict(enc.saturation),projection_saturation=proj_sat,rvq_saturation=rvq_sat,
                   indices_decode_exact=True,integer_audio_metrics=audio_metrics(pcm[:length]/32768,decoded[:length],sr),
                   reference_audio_metrics=audio_metrics(pcm[:length]/32768,reference[:length],sr),
                   vqout_nmse=nmse(y.astype(float)*rvq.manifest['output_scale'],reference_y.cpu().numpy()[0]))
        row['integer_audio_metrics'].update(waveform_metrics(pcm[:length]/32768,decoded[:length]))
        row['reference_audio_metrics'].update(waveform_metrics(pcm[:length]/32768,reference[:length]))
        row['aligned_si_sdr_delta'] = row['integer_audio_metrics']['aligned_si_sdr']-row['reference_audio_metrics']['aligned_si_sdr']
        sf.write(destination/'original.wav',pcm[:length]/32768,sr,subtype='FLOAT')
        sf.write(destination/'qat_fp_codebook.wav',reference[:length],sr,subtype='FLOAT')
        sf.write(destination/'integer_offline.wav',decoded[:length],sr,subtype='FLOAT')
        np.savez(destination/'integer_latents.npz',encoder=encoded,query=query,rvq_output=y,indices=index)
        if args.mode == 'compare':
            enc.reset()
            decoder_state = None
            outputs = []
            offsets = {name:0 for name in trace}
            for frame,start_frame in enumerate(range(0,len(pcm),320)):
                ei,es,tr = enc.run(pcm[start_frame:start_frame+320])
                for name,value in tr.items():
                    pos = offsets[name]
                    np.testing.assert_array_equal(value,trace[name][:,pos:pos+value.shape[-1]],err_msg=f'frame {frame} layer {name}')
                    offsets[name] += value.shape[-1]
                np.testing.assert_array_equal(ei,encoded[:,frame:frame+1])
                qi,_ = project_integer(ei.T,proj)
                yi,ii,_ = rvq.encode(qi)
                np.testing.assert_array_equal(qi,query[frame:frame+1])
                np.testing.assert_array_equal(ii,index[frame:frame+1])
                np.testing.assert_array_equal(yi,y[frame:frame+1])
                dec_y,_ = rvq.decode(ii)
                latent = model.rq_output_projection(latent_tensor(dec_y)).transpose(1,2)
                recon,decoder_state = stream_module(model.decoder,latent,decoder_state)
                if recon.shape[-1] != 320:
                    raise ValueError('Streaming Decoder did not produce 320 samples')
                outputs.append(recon.cpu().numpy().reshape(-1))
                if (frame+1)%50 == 0:
                    print(f'  streamed {frame+1}/{len(pcm)//320} frames',flush=True)
            for name,pos in offsets.items():
                if pos != trace[name].shape[-1]:
                    raise ValueError(f'Unconsumed layer trace: {name}')
            streamed = np.concatenate(outputs)
            exact = bool(np.allclose(streamed,decoded,atol=args.decoder_atol,rtol=args.decoder_rtol))
            sf.write(destination/'integer_stream.wav',streamed[:length],sr,subtype='FLOAT')
            row.update(integer_stream_exact=True,decoder_stream_close=exact,
                       decoder_stream_max_abs_error=float(np.max(np.abs(streamed-decoded))),
                       stream_audio_metrics=audio_metrics(pcm[:length]/32768,streamed[:length],sr))
        rows.append(row)
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(f'[{i+1}/{len(selected)}] saved {destination}',flush=True)
        if row.get('decoder_stream_close') is False:
            raise RuntimeError('Decoder streaming mismatch; WAVs and report saved, no pass claimed')
    deltas = [r['aligned_si_sdr_delta'] for r in rows]
    report.update(software_checks_passed=True,summary=dict(
        num_files=len(rows),mean_aligned_si_sdr_delta=float(np.mean(deltas)),
        p10_aligned_si_sdr_delta=float(np.percentile(deltas,10)),
        worst_aligned_si_sdr_delta=float(np.min(deltas)),
        mean_vqout_nmse=float(np.mean([r['vqout_nmse'] for r in rows]))))
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Software diagnostic complete. No training, deployment approval or RTL verification.',flush=True)


if __name__ == '__main__':
    main()
