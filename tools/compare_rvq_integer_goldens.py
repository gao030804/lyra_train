"""Compare a simulator-produced NPZ dump against one RVQ integer golden."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(expected, actual):
    required=['indices','output_int16']
    required += sorted(k for k in expected if k.startswith('q') and
                       k.endswith(('_residual','_scores','_after_subtract','_stored_residual')))
    if len(required)==2:
        raise ValueError('Golden must include per-stage residual/score/subtraction arrays')
    missing=[k for k in required if k not in expected or k not in actual]
    if missing:
        raise ValueError(f'Missing comparison arrays: {missing}')
    counts={}
    for key in required:
        a,b=expected[key],actual[key]
        if a.shape!=b.shape:
            raise ValueError(f'{key}: shape mismatch {a.shape} vs {b.shape}')
        if a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu':
            raise ValueError(f'{key}: integer payloads required')
        counts[key]=int(np.count_nonzero(a!=b))
    return dict(passed=all(v==0 for v in counts.values()),mismatch_counts=counts,
                rtl_index_mismatch_count=counts['indices'],rtl_latent_mismatch_count=counts['output_int16'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--golden',type=Path,required=True)
    p.add_argument('--rtl-dump',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if args.golden.resolve()==args.rtl_dump.resolve():
        raise ValueError('RTL dump must be a separate simulator-produced artifact')
    with np.load(args.golden,allow_pickle=False) as a, np.load(args.rtl_dump,allow_pickle=False) as b:
        result=compare(a,b)
    result.update(golden=str(args.golden.resolve()),rtl_dump=str(args.rtl_dump.resolve()),
                  scope='Only supplied arrays; simulator provenance must be verified externally')
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
