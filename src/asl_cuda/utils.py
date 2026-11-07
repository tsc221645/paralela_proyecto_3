import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
import numpy as np
import psutil
import torch
import yaml

def config(path):
    c = yaml.safe_load(Path(path).read_text())
    for key in ('batch_size', 'epochs', 'repeats', 'cpu_threads'):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f'{key} debe ser entero positivo')
    if c['num_workers'] < 0 or c['warmup_batches'] < 0:
        raise ValueError('workers y warmup deben ser no negativos')
    if c['precision'] != 'float32':
        raise ValueError('La comparación principal exige float32')
    return c

def seed_all(seed, threads):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

def device_for(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible. Instale PyTorch CUDA y use una GPU NVIDIA; no se hará fallback CPU.')
    return torch.device('cuda:0' if name == 'cuda' else 'cpu')

def hardware():
    try:
        smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        smi = 'nvidia-smi no disponible'
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'sin_commit'
    digest = hashlib.sha256()
    for p in sorted(Path('src').rglob('*.py')):
        digest.update(str(p).encode()); digest.update(p.read_bytes())
    return dict(python=platform.python_version(), platform=platform.platform(),
                cpu=platform.processor(), cores=psutil.cpu_count(), ram_gb=psutil.virtual_memory().total/2**30,
                torch=torch.__version__, numpy=np.__version__, cuda_build=torch.version.cuda,
                cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                nvidia_smi=smi, commit=commit, source_sha256=digest.hexdigest())

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
