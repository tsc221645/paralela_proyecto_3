"""Valida particiones congeladas sin descargar ni sustituir datos reales."""
import _bootstrap
import argparse
import hashlib
from pathlib import Path
from asl_cuda.dataset import datasets
from asl_cuda.utils import config, write_json
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/benchmark.yaml')
    a = p.parse_args()
    c = config(a.config)
    if c['synthetic']: raise ValueError('Este comando valida ASL real')
    parts = datasets(c)
    seen = set()
    for data in parts:
        names = {str(path) for path, _ in data.samples}
        if len(names) != len(data): raise ValueError('Muestras duplicadas')
        if names & seen: raise ValueError('Fuga entre particiones')
        seen |= names
        for i in range(len(data)):
            x, y = data[i]
            assert x.shape == (3,64,64) and x.device.type=='cpu' and 0<=x.min()<=x.max()<=1
    write_json('results/evidence/dataset_validation.json',
               dict(counts=[len(x) for x in parts], manifests=[x.manifest_hash for x in parts], status='OK'))
    print('Dataset validado; 29 clases; particiones disjuntas')
