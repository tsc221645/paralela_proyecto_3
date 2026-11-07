"""Agregación estricta de una sesión; nunca mezcla máquinas o configuraciones."""
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev, median

PHASES = ['preprocess_s','train_s','eval_s','total_s']

def aggregate(directory):
    directory = Path(directory)
    groups, metadata = {}, {}
    for device in ('cpu','cuda'):
        path = directory/f'{device}_runs.csv'
        if not path.exists(): continue
        rows = list(csv.DictReader(path.open()))
        if not rows: raise ValueError('CSV vacío')
        if len(rows) < 5: raise ValueError('Se requieren al menos cinco repeticiones')
        meta = json.loads((directory/f'{device}_metadata.json').read_text())
        for row in rows:
            for key in ('config_sha256','dataset_sha256'):
                if row[key] != meta[key]: raise ValueError('CSV y metadata incompatibles')
            for key in PHASES+['h2d_s','d2h_s','loss','accuracy']:
                value=float(row[key])
                if not math.isfinite(value) or value<0: raise ValueError('Valor no válido')
            if not 0<=float(row['accuracy'])<=1: raise ValueError('accuracy fuera de rango')
        if len({r['run_id'] for r in rows}) != len(rows): raise ValueError('Run duplicado')
        groups[device]=rows; metadata[device]=meta
    if not groups: raise ValueError('No hay resultados en la carpeta indicada')
    if len(metadata)==2:
        for key in ('config_sha256','dataset_sha256'):
            if metadata['cpu'][key] != metadata['cuda'][key]: raise ValueError('Comparación no equivalente')
        for key in ('platform','cpu','cores','ram_gb','torch','source_sha256','gpu'):
            if metadata['cpu']['hardware'][key] != metadata['cuda']['hardware'][key]:
                raise ValueError(f'Máquina/código distintos: {key}')
    summary=[]
    for device, rows in groups.items():
        for phase in PHASES+['h2d_s','d2h_s','gpu_compute_s','accuracy','loss']:
            values=[float(r[phase]) for r in rows]
            summary.append(dict(device=device,phase=phase,n=len(values),mean=mean(values),
                                std=stdev(values),median=median(values)))
    speedups=[]
    if len(groups)==2:
        for phase in PHASES:
            cpu=mean(float(r[phase]) for r in groups['cpu'])
            gpu=mean(float(r[phase]) for r in groups['cuda'])
            if gpu<=0: raise ValueError('Tiempo CUDA no positivo')
            speedups.append(dict(phase=phase,speedup=cpu/gpu,reduction_percent=100*(1-gpu/cpu)))
    return groups,metadata,summary,speedups

def write_csv(path, rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
