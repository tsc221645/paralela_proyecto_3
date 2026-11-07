from pathlib import Path
import copy
import csv
import json
import pytest
import torch
from asl_cuda.dataset import datasets
from asl_cuda.train import execute
from asl_cuda.utils import config
from asl_cuda.reporting import aggregate

def test_benchmark_cpu_real_steps_and_determinism(tmp_path):
    c=config('configs/smoke.yaml')
    a,h=execute(c,torch.device('cpu'),*datasets(c),checkpoint=tmp_path/'model.pt')
    ckpt=torch.load(tmp_path/'model.pt',weights_only=True)
    assert ckpt['metrics']['loss']==a['loss'] and len(ckpt['classes'])==29
    b,_=execute(c,torch.device('cpu'),*datasets(c))
    assert a['loss']==b['loss'] and a['accuracy']==b['accuracy']
    assert a['train_s']>0 and a['eval_s']>0 and a['preprocess_s']>0
    assert a['h2d_s']==a['d2h_s']==0 and a['total_s']>=a['train_s']+a['eval_s']
    assert len(h)==c['epochs'] and 0<=a['accuracy']<=1

def test_aggregation_rejects_incomparable_data(tmp_path):
    fields=['run_id','config_sha256','dataset_sha256','preprocess_s','train_s','eval_s',
            'total_s','h2d_s','d2h_s','loss','accuracy','gpu_compute_s']
    hw=dict(platform='same',cpu='same',cores=1,ram_gb=4,torch='x',source_sha256='x',gpu='gpu')
    for device in ('cpu','cuda'):
        meta=dict(config_sha256='a',dataset_sha256=device,hardware=hw)
        (tmp_path/f'{device}_metadata.json').write_text(json.dumps(meta))
        with (tmp_path/f'{device}_runs.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for i in range(5):
                r={k:1 for k in fields};r.update(run_id=str(i),config_sha256='a',dataset_sha256=device)
                w.writerow(r)
    with pytest.raises(ValueError,match='no equivalente'):aggregate(tmp_path)

def test_speedup_statistics_with_known_measurements(tmp_path):
    fields=['run_id','config_sha256','dataset_sha256','preprocess_s','train_s','eval_s',
            'total_s','h2d_s','d2h_s','loss','accuracy','gpu_compute_s','peak_gpu_mem_mb']
    hw=dict(platform='same',cpu='same',cores=1,ram_gb=4,torch='x',source_sha256='x',gpu='gpu')
    for device, factor in [('cpu',1),('cuda',.5)]:
        (tmp_path/f'{device}_metadata.json').write_text(json.dumps(dict(config_sha256='a',dataset_sha256='b',hardware=hw)))
        with (tmp_path/f'{device}_runs.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for i in range(1,6):
                r={k:i*factor for k in fields}
                r.update(run_id=str(i),config_sha256='a',dataset_sha256='b',accuracy=.5)
                w.writerow(r)
    _,_,summary,speedups=aggregate(tmp_path)
    total=next(r for r in summary if r['device']=='cpu' and r['phase']=='total_s')
    assert total['mean']==total['median']==3
    assert total['std']==pytest.approx(2.5**.5)
    assert all(r['speedup']==2 and r['reduction_percent']==50 for r in speedups)
