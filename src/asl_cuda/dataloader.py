import random
import numpy as np
import torch
from torch.utils.data import DataLoader

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

def loader(dataset, config, shuffle=False):
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle,
                      num_workers=config['num_workers'], pin_memory=config['pin_memory'],
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(config['seed']))
