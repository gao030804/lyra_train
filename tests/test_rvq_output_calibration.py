import unittest
import numpy as np
from tools.rvq_output_calibration import raw_output,output_diagnostics,calibrate_output_scale
from tools.integer_rvq_reference import run_integer,lookup


class OutputCalibrationTests(unittest.TestCase):
    def test_calibration_removes_clip_without_changing_indices(self):
        books=[np.array([[0,0],[120,-120]],dtype=np.int8)]*2
        scales=[.1,.1]
        query=np.array([[[300,-300],[10,-10]]],dtype=np.int16)
        old_scale=.0001
        old,indices,trace=run_integer(query,books,scales,old_scale)
        stats=output_diagnostics(raw_output(indices,books,scales,old_scale),old_scale)
        self.assertGreater(stats['saturation_count'],0)
        self.assertGreater(stats['max_excess_integer'],0)
        new_scale,report=calibrate_output_scale([indices],books,scales,old_scale)
        self.assertGreater(new_scale,old_scale)
        self.assertEqual(report['calibration_output_saturation'],0)
        new,new_indices,new_trace=run_integer(query,books,scales,new_scale)
        np.testing.assert_array_equal(indices,new_indices)
        for a,b in zip(trace,new_trace):
            np.testing.assert_array_equal(a['scores'],b['scores'])
            np.testing.assert_array_equal(a['after_subtract'],b['after_subtract'])
        np.testing.assert_array_equal(new,lookup(indices,books,scales,new_scale))
        self.assertEqual(output_diagnostics(raw_output(indices,books,scales,new_scale),new_scale)['saturation_count'],0)

    def test_diagnostics_signed_limits_and_empty_calibration(self):
        raw=np.array([[32767,32768,-32768,-32770]],dtype=np.int64)
        d=output_diagnostics(raw,.01)
        self.assertEqual(d['saturation_count'],2)
        self.assertEqual(d['max_excess_integer'],2)
        self.assertEqual(d['examples'][0]['position'],[0,1])
        with self.assertRaises(ValueError):
            calibrate_output_scale([],[],[],.01)

    def test_never_shrinks_existing_output_scale(self):
        books=[np.array([[1,-1]],dtype=np.int8)]
        indices=np.zeros((1,2,1),dtype=np.int64)
        scale,report=calibrate_output_scale([indices],books,[.01],.1)
        self.assertEqual(scale,.1)
        self.assertEqual(report['calibration_segments'],1)


if __name__=='__main__':
    unittest.main()
