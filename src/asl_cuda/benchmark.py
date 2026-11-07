import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from .dataset import datasets, data_fingerprint
from .evidence import Monitor
from .train import execute, warmup
from .utils import config, device_for, hardware, write_json

def main(default_device=None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/benchmark.yaml')
    p.add_argument('--device', choices=['cpu','cuda'], default=default_device or 'cpu')
    p.add_argument('--repeats', type=int)
    p.add_argument('--session', default=None)
    args = p.parse_args()
    c = config(args.config)
    if args.repeats is not None:
        if args.repeats < 1: p.error('repeats debe ser positivo')
        c['repeats'] = args.repeats
    device = device_for(args.device)
    session = args.session or uuid4().hex[:12]
    directory = Path(c['output']) / 'raw' / session
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f'{args.device}_runs.csv'
    if csv_path.exists():
        raise FileExistsError(f'No se sobrescribe {csv_path}; use otra sesión')
    start = perf_counter()
    train, validation, test = datasets(c)
    # Huella de contenido antes del benchmark; no basta con nombres de imágenes.
    fingerprint = data_fingerprint((train, validation, test))
    setup_s = perf_counter() - start
    signature = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
    metadata = dict(config=c, hardware=hardware(), dataset_sha256=fingerprint,
                    config_sha256=signature, setup_s=setup_s,
                    classes=29, counts=[len(train),len(validation),len(test)])
    write_json(directory / f'{args.device}_metadata.json', metadata)
    start = perf_counter()
    warmup(c, device)
    metadata['warmup_s'] = perf_counter() - start
    write_json(directory / f'{args.device}_metadata.json', metadata)
    for i in range(c['repeats']):
        run_id = f'{args.device}_{i+1}'
        with Monitor(Path(c['output'])/'evidence'/session/f'{run_id}.csv',
                     gpu_id=metadata['hardware'].get('gpu_uuid') or 0):
            result, history = execute(c, device, train, validation, test,
                                      checkpoint=directory/f'{args.device}_model.pt' if i==c['repeats']-1 else None)
        row = dict(run_id=run_id, variant='cuda' if args.device=='cuda' else 'refactored_cpu',
                   synthetic=c['synthetic'], seed=c['seed'], n_images=len(train),
                   batch_size=c['batch_size'], epochs=c['epochs'], num_workers=c['num_workers'],
                   dataset_sha256=fingerprint, config_sha256=signature, **result)
        with csv_path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if i == 0: writer.writeheader()
            writer.writerow(row)
        write_json(directory/f'{run_id}_epochs.json', history)
        print(json.dumps(row), flush=True)
    print(f'Resultados: {csv_path}')
