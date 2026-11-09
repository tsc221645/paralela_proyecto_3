"""Variantes en procesos separados para aislar memoria y estado del backend."""
import argparse
import subprocess
import sys
from uuid import uuid4
from pathlib import Path
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/benchmark.yaml')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--devices', nargs='+', choices=['cpu','cuda'], default=['cpu','cuda'])
    a = p.parse_args()
    if 'cuda' in a.devices:
        import _bootstrap
        from asl_cuda.utils import device_for
        device_for('cuda')
    session = uuid4().hex[:12]
    for device in a.devices:
        subprocess.run([sys.executable, f'scripts/run_{device}.py', '--config', a.config,
                        '--repeats', str(a.repeats), '--session', session], check=True)
    benchmark_config = __import__('yaml').safe_load(Path(a.config).read_text())
    subprocess.run([sys.executable, 'scripts/plot_results.py', '--input',
                    str(Path(benchmark_config['output'])/'raw'/session),
                    '--min-repeats', str(benchmark_config['repeats'])], check=True)
