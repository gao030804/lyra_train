import json
import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.integer_codec_package import BinaryEncoder, BinaryRVQ, unpack_weight, wide_requant
from tools.integer_rvq_reference import export_package, run_integer


def packed(w):
    co,ci,k = w.shape
    a = np.zeros(((co+7)//8,(ci*k+3)//4,4,8),dtype=np.int8)
    for o in range(co):
        for t in range(k):
            for c in range(ci):
                j = t*ci+c
                a[o//8,j//4,j%4,o%8] = w[o,c,t]
    return a.tobytes()


def encoder_fixture(root):
    weights = [np.array([[[1,2,-1]],[[2,0,1]]],dtype=np.int8),
               np.array([[[1,-1,2]],[[2,1,-1]]],dtype=np.int8),
               np.array([[[1],[2]],[[0],[1]]],dtype=np.int8)]
    wb,bb,pb = bytearray(),bytearray(),bytearray()
    layers = []
    for i,w in enumerate(weights):
        co,ci,k = w.shape
        layer = dict(name=str(i),cin=1 if i==0 else 2,cout=co,kernel=k,
                     stride=2 if i==0 else 1,dilation=2 if i==1 else 1,
                     groups=2 if i==1 else 1,weight_cin_per_group=ci,
                     left_pad=(2 if i==1 else 1)*(k-1)+1-(2 if i==0 else 1),
                     input_scale=1/32768,output_scale=1/32768,
                     weight_offset=len(wb),weight_bytes=len(packed(w)),bias_offset=len(bb),parameter_offset=len(pb))
        layers.append(layer)
        wb.extend(packed(w))
        bb.extend(np.array([1,-1]+[0]*6,dtype='<i8').tobytes())
        pb.extend(np.array([1,1]+[0]*6,dtype='<i4').tobytes()+bytes(16))
    residual = dict(name='res',identity_scale=1/32768,branch_scale=1/32768,output_scale=1/32768,
                    alignment={k:dict(multiplier=1,shift=0) for k in ('identity','branch','gain')})
    m = dict(weight_bits=8,activation_bits=16,accumulator_bits=40,rounding='half_away_from_zero',
             zero_point=0,bias_file='bias.bin',layers=layers,residuals=[residual])
    (root/'encoder_manifest.json').write_text(json.dumps(m))
    (root/'encoder_weights_int8.bin').write_bytes(wb)
    (root/'bias.bin').write_bytes(bb)
    (root/'encoder_quant_params.bin').write_bytes(pb)
    graph = dict(kind='seq',children=[dict(kind='conv',name='0'),
                 dict(kind='relu',name='relu',scale=1/32768),
                 dict(kind='residual',name='res',branch=dict(kind='seq',children=[dict(kind='conv',name='1'),dict(kind='conv',name='2')]))])
    return graph,weights


class IntegerCodecTests(unittest.TestCase):
    def test_pack_layout_roundtrip(self):
        w = np.arange(11*3*5,dtype=np.int64).reshape(11,3,5)%127
        np.testing.assert_array_equal(unpack_weight(packed(w),11,3,5),w)

    def test_wide_signed_half_rounding(self):
        x = np.array([-2**39+1,2**39-1],dtype=np.int64)
        m,s = 2**31-1,35
        expected = [(-1 if v<0 else 1)*((abs(int(v)*m)+(1<<(s-1)))>>s) for v in x]
        np.testing.assert_array_equal(wide_requant(x,m,s),expected)
        np.testing.assert_array_equal(wide_requant(np.array([-3,-1,1,3]),1,1),[-2,-1,1,2])

    def test_stream_history_and_reset_dense_depthwise_residual(self):
        with tempfile.TemporaryDirectory() as d:
            graph,_ = encoder_fixture(Path(d))
            enc = BinaryEncoder(d,graph)
            pcm = np.random.default_rng(4).integers(-20,20,32,dtype=np.int16)
            whole,_,tr = enc.run(pcm)
            enc.reset()
            pieces, traces = [], {k:[] for k in tr}
            for start in range(0,len(pcm),4):
                y,_,t = enc.run(pcm[start:start+4])
                pieces.append(y)
                for k,v in t.items():
                    traces[k].append(v)
            np.testing.assert_array_equal(np.concatenate(pieces,-1),whole)
            for k in tr:
                np.testing.assert_array_equal(np.concatenate(traces[k],-1),tr[k])
            enc.reset()
            np.testing.assert_array_equal(enc.run(pcm)[0],whole)

    def test_conv_matches_scalar_oracle(self):
        with tempfile.TemporaryDirectory() as d:
            graph,weights = encoder_fixture(Path(d))
            enc = BinaryEncoder(d,graph)
            x = np.arange(24,dtype=np.int16).reshape(2,12)-10
            for name in ('1','2'):
                a = enc.layers[name]
                enc.reset()
                actual = enc.conv(name,x)
                expected = np.zeros_like(actual)
                w = weights[int(name)]
                for o in range(a['cout']):
                    for t in range(actual.shape[-1]):
                        acc = int(a['b'][o])
                        for c in range(w.shape[1]):
                            channel = (o//(a['cout']//a['groups']))*w.shape[1]+c
                            for k in range(w.shape[2]):
                                pos = t*a['stride']-a['left_pad']+k*a['dilation']
                                if pos>=0:
                                    acc += int(x[channel,pos])*int(w[o,c,k])
                        expected[o,t] = np.clip(acc,-32768,32767)
                np.testing.assert_array_equal(actual,expected)

    def test_stride_rejects_partial_and_tracks_saturation(self):
        with tempfile.TemporaryDirectory() as d:
            graph,_ = encoder_fixture(Path(d))
            enc = BinaryEncoder(d,graph)
            with self.assertRaises(ValueError):
                enc.run(np.zeros(3,dtype=np.int16))
            enc.reset()
            enc.run(np.full(32,32767,dtype=np.int16))
            self.assertGreater(sum(enc.saturation.values()),0)

    def test_rvq_binary_and_indices_decode(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(8)
            books = [rng.integers(-30,30,(k,4),dtype=np.int8) for k in (16,8,8)]
            scales = [.03,.02,.01]
            export_package(d,books,scales,.0003,provenance={})
            rvq = BinaryRVQ(d)
            query = rng.integers(-100,100,(20,4),dtype=np.int16)
            out,indices,_ = rvq.encode(query)
            expected,ei,_ = run_integer(query,books,scales,.0003)
            np.testing.assert_array_equal(out,expected)
            np.testing.assert_array_equal(indices,ei)
            np.testing.assert_array_equal(rvq.decode(indices)[0],out)
            blob = Path(d)/'rvq_codebook_int8.bin'
            blob.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                BinaryRVQ(d)

    def test_projection_export_readback(self):
        from tools.rvq_scale_qat import make_projection, export_projection, project_integer
        from tools.reconstruct_integer_codec import load_projection
        rng = np.random.default_rng(12)
        spec = make_projection(rng.normal(size=(3,5)),np.zeros(3),.0001,.001)
        x = rng.integers(-300,300,(5,5),dtype=np.int16)
        with tempfile.TemporaryDirectory() as d:
            export_projection(d,spec)
            restored = load_projection(d)
            np.testing.assert_array_equal(project_integer(x,spec)[0],project_integer(x,restored)[0])

    def test_actual_decoder_transpose_overlap_state(self):
        # Execute the actual repository class, without importing optional AudioLM
        # dependencies. This is a component test, not a full checkpoint run.
        import torch
        import torch.nn.functional as F
        source = (Path(__file__).resolve().parents[1]/'audiolm_pytorch/soundstream.py').read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='CausalConvTranspose1d')
        ns = dict(torch=torch,nn=torch.nn,Module=torch.nn.Module,F=F,
                  exists=lambda x:x is not None,rearrange=lambda x,pattern:x.reshape(1,-1,1))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-decoder-component>','exec'),ns)
        torch.manual_seed(42)
        layer = ns['CausalConvTranspose1d'](4,3,8,4).eval()
        x = torch.randn(1,4,9)
        with torch.no_grad():
            full = layer(x)
            state = None
            parts = []
            for frame in x.split(1,dim=-1):
                y,state = layer.forward_stream(frame,state)
                parts.append(y)
            torch.testing.assert_close(torch.cat(parts,-1),full,atol=1e-6,rtol=1e-5)
            reset_every_frame = torch.cat([layer.forward_stream(f,None)[0] for f in x.split(1,dim=-1)],-1)
            self.assertGreater(float((reset_every_frame-full).abs().max()),.01)

    def test_report_selection_preserves_crops(self):
        from types import SimpleNamespace
        from tools.reconstruct_integer_codec import jobs
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'report.json'
            rows = [dict(path=f'{n}.flac',start_sample=320*n) for n in range(10)]
            path.write_text(json.dumps(dict(per_file=rows)))
            args = SimpleNamespace(audio=None,report=path,file_index=2,num_files=1)
            self.assertEqual(jobs(args),[rows[2]])


if __name__ == '__main__':
    unittest.main()
