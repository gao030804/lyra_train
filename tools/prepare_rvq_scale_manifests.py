"""Create reproducible, speaker-disjoint scale-QAT lists from LibriSpeech."""
import argparse
import csv
from pathlib import Path
import random
import soundfile as sf
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train-count',type=int,default=32)
    p.add_argument('--validation-count',type=int,default=20)
    p.add_argument('--test-count',type=int,default=10)
    p.add_argument('--min-seconds',type=float,default=4.)
    a=p.parse_args()
    root=a.audio_dir.resolve()
    speakers=sorted(x for x in root.iterdir() if x.is_dir() and any(x.rglob('*.flac')))
    order=torch.randperm(len(speakers),generator=torch.Generator().manual_seed(a.seed)).tolist()
    speakers=[speakers[i] for i in order]
    ntrain=round(.9*len(speakers)); nval=round(.05*len(speakers))
    splits=[speakers[:ntrain],speakers[ntrain:ntrain+nval],speakers[ntrain+nval:]]
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for label,group,count in zip(('train','validation','test'),splits,(a.train_count,a.validation_count,a.test_count)):
        files=sorted(path for speaker in group for path in speaker.rglob('*.flac'))
        if label=='test':
            files=sorted(files,key=lambda path:(-sf.info(str(path)).duration,str(path)))
        else:
            random.Random(a.seed).shuffle(files)
        chosen=[]
        for path in files:
            if sf.info(str(path)).duration >= a.min_seconds:
                chosen.append(path)
            if len(chosen)==count:
                break
        if count<=0 or len(chosen)!=count:
            raise ValueError(f'Insufficient audio for {label}, count={count}')
        dest=a.output_dir/f'{label}.csv'
        with dest.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['source'])
            writer.writeheader()
            writer.writerows(dict(source=str(path)) for path in chosen)
        print(f'{label}: {dest} ({count} files)')


if __name__=='__main__':
    main()
