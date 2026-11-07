"""Lectura, resize bilineal y normalización exclusivamente en CPU."""
import csv
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate

CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['del', 'nothing', 'space']

class ASLDataset(Dataset):
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.rows = list(csv.DictReader(Path(manifest).open(encoding='utf-8')))
        if not self.rows:
            raise ValueError('Partición vacía')
        self.samples = []
        for row in self.rows:
            label = CLASSES.index(row['class'])
            name = Path(row['filename']).name
            if name != row['filename']:
                raise ValueError('filename debe ser un nombre simple')
            path = self.root / row['class'] / name
            if not path.is_file():
                raise FileNotFoundError(f'Falta imagen {row["class"]}/{name}; configure data_root')
            self.samples.append((path, label))
        self.manifest_hash = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            array = np.array(image.convert('RGB'), dtype=np.float32)
        x = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        # TF notebook: resize bilinear float32, sin antialias; no PIL resize.
        x = interpolate(x, size=(64, 64), mode='bilinear', align_corners=False,
                        antialias=False)[0] / 255.0
        return x.contiguous(), label

class SyntheticDataset(Dataset):
    """Solo verifica ejecución; sus métricas no representan reconocimiento ASL."""
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 3, 64, 64, generator=g)
        self.y = torch.arange(n) % len(CLASSES)
        self.manifest_hash = f'synthetic:{n}:{seed}'
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def datasets(config):
    if config.get('synthetic', False):
        return (SyntheticDataset(config['synthetic_train'], config['seed']),
                SyntheticDataset(config['synthetic_eval'], config['seed'] + 1),
                SyntheticDataset(config['synthetic_eval'], config['seed'] + 2))
    directory = Path(config['splits_dir'])
    return tuple(ASLDataset(config['data_root'], directory / f'{s}_split.csv')
                 for s in ('train', 'validation', 'test'))


def data_fingerprint(parts):
    digest = hashlib.sha256()
    for data in parts:
        digest.update(data.manifest_hash.encode())
        for path, _ in getattr(data, 'samples', []):
            digest.update(path.read_bytes())
    return digest.hexdigest()
