from contextlib import contextmanager
from time import perf_counter
import torch

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

@contextmanager
def timed(times, key, device, events=None):
    sync(device)
    start = perf_counter()
    if events is not None and device.type == 'cuda':
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
    yield
    if events is not None and device.type == 'cuda':
        b.record()
    sync(device)
    times[key] += perf_counter() - start
    if events is not None and device.type == 'cuda':
        events.append(a.elapsed_time(b) / 1000)
