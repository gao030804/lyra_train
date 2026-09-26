import unittest
import tempfile
from pathlib import Path
import struct
import numpy as np
import torch
from tools.rvq_scale_qat import ScaleRVQ,make_projection,project_integer,retention_summary
from tools.integer_rvq_reference import run_integer,lookup,export_package,ratio_parameters


class ScaleQATTests(unittest.TestCase):
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
        self.assertGreater(float(model.delta.abs().sum()),0)
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
