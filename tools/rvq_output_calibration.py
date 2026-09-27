"""Train/calibration-only output-range calibration with frozen RVQ indices."""
import numpy as np
from tools.integer_rvq_reference import requant, ratio_parameters


def raw_output(indices, books, scales, output_scale):
    total=sum(requant(book[indices[...,q]].astype(np.int64),*ratio_parameters(float(scale)/output_scale))
              for q,(book,scale) in enumerate(zip(books,scales)))
    if np.any((total < -(2**31)) | (total > 2**31-1)):
        raise OverflowError('RVQ output accumulation exceeds INT32')
    return total


def output_diagnostics(raw, scale):
    mask=(raw < -32768)|(raw > 32767)
    positions=np.argwhere(mask)
    excess=np.maximum(raw-32767,-32768-raw).clip(min=0)
    return dict(output_scale=float(scale),preclip_min=int(raw.min()),preclip_max=int(raw.max()),
                saturation_count=int(mask.sum()),max_excess_integer=int(excess.max()),
                preclip_abs_peak_real=float(np.abs(raw).max()*scale),
                examples=[dict(position=p.tolist(),preclip=int(raw[tuple(p)]),
                               excess_integer=int(excess[tuple(p)])) for p in positions[:16]])


def calibrate_output_scale(index_batches, books, scales, old_scale, headroom=1.05):
    if not index_batches or not np.isfinite(headroom) or headroom<1 or not np.isfinite(old_scale) or old_scale<=0:
        raise ValueError('Nonempty calibration data, positive scale and headroom >=1 required')
    peak=max(float(np.abs(sum(book[idx[...,q]].astype(float)*scale
             for q,(book,scale) in enumerate(zip(books,scales)))).max()) for idx in index_batches)
    # Reserve a half-unit per stage plus one unit for output requant rounding.
    denominator=32767-len(books)*.5-1
    if denominator<=0:
        raise ValueError('Too many quantizer stages')
    new_scale=max(old_scale,headroom*peak/denominator)
    for _ in range(16):
        stats=[output_diagnostics(raw_output(idx,books,scales,new_scale),new_scale) for idx in index_batches]
        if sum(s['saturation_count'] for s in stats)==0:
            return new_scale,dict(old_output_scale=float(old_scale),new_output_scale=float(new_scale),
                headroom=float(headroom),calibration_unrounded_peak=peak,
                calibration_output_saturation=0,calibration_segments=len(index_batches))
        new_scale*=1.01
    raise RuntimeError('Could not obtain unsaturated calibration output')
