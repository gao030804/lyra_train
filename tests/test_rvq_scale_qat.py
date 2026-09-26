import unittest
import tempfile
from pathlib import Path
import struct
import numpy as np
import torch
from tools.rvq_scale_qat import ScaleRVQ,make_projection,project_integer,retention_summary,projection_output_scale,audio_first_key
from tools.integer_rvq_reference import run_integer,lookup,export_package,ratio_parameters


class ScaleQATTests(unittest.TestCase):
    def test_experimental_gate_does_not_override_strict(self):
        rows=[dict(aligned_si_sdr_delta=-.01,aligned_corr_delta=0,vqout_nmse=.047,
                   saturation=0,q00=dict(index_flip=.1,energy_weighted_flip=.02)) for _ in range(10)]
        strict=retention_summary(rows)
        experiment=retention_summary(rows,'experimental')
        self.assertFalse(strict['passed'])
        self.assertTrue(experiment['passed'])
        self.assertFalse(experiment['strict_pass'])
        self.assertTrue(experiment['audio_pass'])
        self.assertAlmostEqual(experiment['q00_index_flip'],.1)
        self.assertAlmostEqual(experiment['q00_energy_weighted_flip'],.02)
        rows[0]['saturation']=1
        self.assertFalse(retention_summary(rows,'experimental')['passed'])

    def test_accumulated_batch_matches_mean_gradient(self):
        rng=np.random.default_rng(21)
        books=[rng.normal(size=(8,4)) for _ in range(2)]
        a=ScaleRVQ(books,[.02,.01],.001,max_ratio=1.08)
        b=ScaleRVQ(books,[.02,.01],.001,max_ratio=1.08)
        queries=[torch.tensor(rng.normal(size=(1,8,4))) for _ in range(4)]
        losses=[]
        for x in queries:
            y,d,c=a(x,x)
            losses.append(y.square().mean()+.1*d+.02*c)
            y,d,c=b(x,x)
            ((y.square().mean()+.1*d+.02*c)/len(queries)).backward()
        torch.stack(losses).mean().backward()
        torch.testing.assert_close(a.delta.grad,b.delta.grad)

    def test_step_zero_baseline_guard(self):
        from tools.train_rvq_quant_scales import verify_ptq_baseline
        row=dict(path='audio.flac',start_sample=100,aligned_si_sdr_delta=.07,
                 aligned_corr_delta=.001,vqout_nmse=.047,saturation=0)
        baseline=dict(per_file=[dict(row,input_saturation=0,residual_saturation=0,output_saturation=0)])
        verify_ptq_baseline(dict(per_file=[row]),baseline)
        with self.assertRaises(ValueError):
            verify_ptq_baseline(dict(per_file=[dict(row,vqout_nmse=.08)]),baseline)
        with self.assertRaises(ValueError):
            verify_ptq_baseline(dict(per_file=[dict(row,start_sample=101)]),baseline)
        with self.assertRaises(ValueError):
            verify_ptq_baseline(dict(per_file=[dict(row,saturation=1)]),baseline)

    def test_only_scales_train_and_receive_gradient(self):
        rng=np.random.default_rng(6)
        books=[rng.normal(size=(16,8)),rng.normal(size=(8,8))]
        model=ScaleRVQ(books,[.023,.02],.001)
        self.assertEqual(sum(p.numel() for p in model.parameters()),2)
        query=torch.tensor(rng.normal(size=(1,20,8)))
        y,d,c=model(query,query)
        loss=y.square().mean()+.2*d+.02*c
        loss.backward()
        self.assertTrue(torch.isfinite(model.delta.grad).all())
        self.assertGreater(float(model.delta.grad.abs().sum()),0)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        optimizer.step()
        self.assertGreater(float(model.delta.detach().abs().sum()),0)
        np.testing.assert_array_equal(model.book_0.numpy(),books[0])
        with torch.no_grad():
            model.delta.copy_(torch.tensor([-100.,100.]))
        torch.testing.assert_close(model.scales(),model.initial*torch.tensor([1/1.15,1.15],dtype=torch.float64))

    def test_fixed_residual_scale_integer_search_and_lookup(self):
        rng=np.random.default_rng(9)
        books=[rng.integers(-128,128,(16,32)).astype(np.int8) for _ in range(3)]
        scales=[.03,.021,.01]
        residual=[.029,.022,.011]
        query=rng.integers(-32768,32768,(1,5,32)).astype(np.int16)
        y,indices,_=run_integer(query,books,scales,.05,residual_scales=residual)
        oracle,oi,_=run_integer(query,books,scales,.05,residual_scales=residual,squared_distance=True)
        np.testing.assert_array_equal(indices,oi)
        np.testing.assert_array_equal(y,oracle)
        np.testing.assert_array_equal(y,lookup(indices,books,scales,.05))

    def test_v1_and_v2_initially_identical(self):
        rng=np.random.default_rng(13)
        books=[rng.normal(size=(16,8)) for _ in range(2)]
        x=torch.tensor(rng.normal(size=(1,10,8)))
        a=ScaleRVQ(books,[.02,.015],.001,scale_mode='v1')
        b=ScaleRVQ(books,[.02,.015],.001,scale_mode='v2')
        torch.testing.assert_close(a(x,x)[0],b(x,x)[0])
        for model in (a,b):
            y,d,c=model(x,x)
            (y.square().mean()+d+c).backward()
            self.assertTrue(torch.isfinite(model.delta.grad).all())
            self.assertGreater(model.delta.grad.abs().sum().item(),0)

    def test_projection_retarget_and_audio_first_selection(self):
        spec=make_projection(np.eye(2),None,.01,.01)
        updated=projection_output_scale(spec,.02)
        x=np.array([[[100,-100]]],dtype=np.int16)
        y,_=project_integer(x,updated)
        np.testing.assert_array_equal(y,[[[50,-50]]])
        np.testing.assert_array_equal(spec['weight'],updated['weight'])
        self.assertEqual(spec['output_scale'],.01)
        a=dict(mean_si_sdr_delta=.1,p10_si_sdr_delta=0,worst_si_sdr_delta=-.1,mean_corr_delta=0,mean_vqout_nmse=.035)
        b=dict(a,mean_si_sdr_delta=0,mean_vqout_nmse=.01)
        self.assertLess(audio_first_key(a),audio_first_key(b))

    def test_projection_identity_and_wide_bias(self):
        spec=make_projection(np.eye(4),None,.01,.01)
        x=np.array([[[123,-123,32767,-32768]]],dtype=np.int16)
        y,sat=project_integer(x,spec)
        np.testing.assert_array_equal(y,x)
        self.assertEqual(sat,0)
        # Product exceeds INT64, yet the requanted value fits INT16.
        spec=dict(weight=np.zeros((1,1),dtype=np.int8),bias=np.array([2**38]),multiplier=np.array([2**30]),shift=np.array([60]))
        y,sat=project_integer(np.array([[0]],dtype=np.int16),spec)
        self.assertEqual(int(y[0,0]),256)

    def test_v2_export_preserves_fixed_residual_interface(self):
        books=[np.zeros((16,32),dtype=np.int8)]*2
        with tempfile.TemporaryDirectory() as directory:
            manifest=export_package(directory,books,[.1,.05],.01,provenance={},residual_scales=[.11,.06])
            self.assertEqual(manifest['format_version'],2)
            self.assertEqual(manifest['input_scale'],.11)
            data=(Path(directory)/'rvq_search_subtract_requant_params.bin').read_bytes()
            self.assertEqual(len(data),20)
            self.assertEqual(struct.unpack('<iBiB',data[:10]),(*ratio_parameters(.11/.1),*ratio_parameters(.1/.11)))

    def test_worst_file_and_saturation_gate(self):
        rows=[dict(aligned_si_sdr_delta=.1,aligned_corr_delta=.01,vqout_nmse=.02,saturation=0) for _ in range(9)]
        rows.append(dict(aligned_si_sdr_delta=-.5,aligned_corr_delta=-.02,vqout_nmse=.02,saturation=0))
        self.assertFalse(retention_summary(rows)['passed'])
        rows[-1]['aligned_si_sdr_delta']=0
        self.assertTrue(retention_summary(rows)['passed'])
        rows[-1]['saturation']=1
        self.assertFalse(retention_summary(rows)['passed'])


if __name__=='__main__':
    unittest.main()
