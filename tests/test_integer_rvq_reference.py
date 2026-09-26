import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.integer_rvq_reference import (
    export_package, lookup, quantize_books, requant, run_integer,
)


class IntegerRVQTests(unittest.TestCase):
    def test_signed_rounding_and_large_shift(self):
        np.testing.assert_array_equal(requant(np.array([-3,-1,1,3]),1,1), [-2,-1,1,2])
        np.testing.assert_array_equal(requant(np.array([-32768,32767]),2**31-1,63), [0,0])

    def test_nonuniform_search_matches_exhaustive_and_decoder(self):
        rng = np.random.default_rng(123)
        for sizes, dim in [([256]+[128]*8,32), ([16]*16,64)]:
            books = [rng.integers(-128,128,(k,dim),dtype=np.int16).astype(np.int8) for k in sizes]
            query = rng.integers(-32768,32768,(1,7,dim),dtype=np.int16)
            scales = [0.1*(.75**q) for q in range(len(books))]
            actual, indices, _ = run_integer(query,books,scales,.1)
            oracle, expected_indices, _ = run_integer(query,books,scales,.1,squared_distance=True)
            np.testing.assert_array_equal(indices, expected_indices)
            np.testing.assert_array_equal(actual, oracle)
            np.testing.assert_array_equal(actual, lookup(indices,books,scales,.1))

    def test_ties_and_17bit_subtraction(self):
        books = [np.array([[-128],[-128]],dtype=np.int8), np.array([[0],[1]],dtype=np.int8)]
        _, indices, trace = run_integer(np.array([[32767]],dtype=np.int16), books,[.1,.2],.1)
        self.assertEqual(indices[0,0],0)
        self.assertEqual(trace[0]['after_subtract'][0,0],32895)
        self.assertEqual(trace[1]['residual'][0,0],16448)

    def test_pytorch_oracle_matches_integer(self):
        from tools.analyze_rvq_codebook_quantization import torch_quantized_oracle
        rng = np.random.default_rng(51)
        books = [rng.integers(-128,128,(k,32),dtype=np.int16).astype(np.int8) for k in (256,128,128)]
        query = rng.integers(-32768,32768,(1,5,32),dtype=np.int16)
        scales=[.03,.017,.008]
        value, index, _ = run_integer(query,books,scales,.07)
        actual, actual_index = torch_quantized_oracle(query,books,scales,.07)
        np.testing.assert_array_equal(value, actual)
        np.testing.assert_array_equal(index, actual_index)

    def test_int32_search_extrema(self):
        books=[np.array([[-128]*128,[127]*128],dtype=np.int8)]
        _,_,trace=run_integer(np.array([[-32768]*128,[32767]*128],dtype=np.int16),books,[1.],1.)
        self.assertTrue((trace[0]['scores'] >= -(2**31)).all())
        self.assertTrue((trace[0]['scores'] <= 2**31-1).all())

    def test_package_packing_norms_and_parameter_sizes(self):
        fp=[np.arange(45,dtype=float).reshape(9,5)-22]
        books=quantize_books(fp,[.2])
        with tempfile.TemporaryDirectory() as tmp:
            export_package(tmp,books,[.2],.1,provenance={})
            root=Path(tmp)
            packed=np.fromfile(root/'rvq_codebook_packed_4x8_int8.bin',dtype=np.int8).reshape(2,2,4,8)
            for k in range(9):
                for d in range(5):
                    self.assertEqual(packed[k//8,d//4,d%4,k%8],books[0][k,d])
            np.testing.assert_array_equal(np.fromfile(root/'rvq_codebook_norm_int32.bin',dtype='<i4'),(books[0].astype(np.int64)**2).sum(-1))
            self.assertEqual((root/'rvq_stage_requant_params.bin').stat().st_size,10)
            manifest=json.loads((root/'rvq_manifest.json').read_text())
            self.assertEqual(manifest['stages'][0]['codebook_size'],9)


if __name__ == '__main__':
    unittest.main()
