"""Instrumentación por fase; sincronización deliberada para atribuir tiempos."""
from collections import defaultdict
from time import perf_counter
import math
import torch
from .dataloader import loader
from .model import BasicCNN
from .timing import timed, sync
from .utils import seed_all

def execute(config, device, train_data, validation, test_data):
    seed_all(config['seed'], config['cpu_threads'])
    times = defaultdict(float)
    sync(device)
    total_start = perf_counter()
    model = BasicCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-7)
    criterion = torch.nn.CrossEntropyLoss()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    history, events = [], []
    for epoch in range(config['epochs']):
        model.train()
        iterator = iter(loader(train_data, config, shuffle=True))
        for _ in range(len(iterator)):
            with timed(times, 'preprocess_s', torch.device('cpu')):
                x, y = next(iterator)
            with timed(times, 'h2d_s' if device.type == 'cuda' else 'cpu_stage_s', device):
                x, y = x.to(device, non_blocking=config['pin_memory']), y.to(device, non_blocking=config['pin_memory'])
            assert x.device == device and next(model.parameters()).device == device
            with timed(times, 'train_s', device, events):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
        metrics = evaluate(model, validation, config, device, criterion, times)
        history.append(dict(epoch=epoch+1, **metrics))
    metrics = evaluate(model, test_data, config, device, criterion, times)
    sync(device)
    total = perf_counter() - total_start
    if not math.isfinite(metrics['loss']):
        raise RuntimeError('Pérdida no finita')
    return dict(**{k: times[k] for k in ('preprocess_s','h2d_s','train_s','eval_s','d2h_s')},
                total_s=total, gpu_compute_s=sum(events),
                peak_gpu_mem_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0,
                model_device=str(next(model.parameters()).device), input_device=str(x.device),
                parameters=sum(p.numel() for p in model.parameters()), **metrics), history

@torch.inference_mode()
def evaluate(model, data, config, device, criterion, times):
    model.eval()
    count, correct, loss_sum = 0, torch.zeros((), device=device), torch.zeros((), device=device)
    iterator = iter(loader(data, config))
    for _ in range(len(iterator)):
        with timed(times, 'preprocess_s', torch.device('cpu')):
            x, y = next(iterator)
        with timed(times, 'h2d_s' if device.type=='cuda' else 'cpu_stage_s', device):
            x, y = x.to(device), y.to(device)
        with timed(times, 'eval_s', device):
            logits = model(x)
            loss_sum += criterion(logits, y) * len(y)
            correct += (logits.argmax(1) == y).sum()
            count += len(y)
    with timed(times, 'd2h_s' if device.type=='cuda' else 'cpu_metrics_s', device):
        values = torch.stack((loss_sum / count, correct / count)).cpu().tolist()
    return dict(loss=values[0], accuracy=values[1])

def warmup(config, device):
    """Contexto CUDA y optimizer calentados fuera de región; modelo descartado."""
    seed_all(config['seed'], config['cpu_threads'])
    model = BasicCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-7)
    x = torch.zeros(config['batch_size'], 3, 64, 64, device=device)
    y = torch.zeros(config['batch_size'], dtype=torch.long, device=device)
    for _ in range(config['warmup_batches']):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    sync(device)
