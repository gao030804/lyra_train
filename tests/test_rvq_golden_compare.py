import unittest
import numpy as np
from tools.compare_rvq_integer_goldens import compare


class GoldenCompareTests(unittest.TestCase):
    def test_exact_and_mismatch(self):
        a={k:np.zeros((2,3),dtype=np.int32) for k in
           ('indices','output_int16','q00_residual','q00_scores','q00_after_subtract')}
        b={k:v.copy() for k,v in a.items()}
        self.assertTrue(compare(a,b)['passed'])
        b['indices'][0,0]=1
        self.assertEqual(compare(a,b)['rtl_index_mismatch_count'],1)
        self.assertFalse(compare(a,b)['passed'])
        del b['q00_scores']
        with self.assertRaises(ValueError):
            compare(a,b)

    def test_shapes_and_float_rejected(self):
        a={'indices':np.zeros(2,dtype=int),'output_int16':np.zeros(2,dtype=int),
           'q00_scores':np.zeros(2,dtype=int)}
        with self.assertRaises(ValueError):
            compare(a,dict(a,indices=np.zeros((1,2),dtype=int)))
        with self.assertRaises(ValueError):
            compare(a,dict(a,indices=np.zeros(2)))
