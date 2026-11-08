"""Verificación local sin dataset ASL; guarda evidencia legible de cada comando."""
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

def run(args, log, expected=0, contains=None):
    result=subprocess.run([sys.executable,*args],capture_output=True,text=True)
    message=('$ python '+' '.join(args)+'\n'+result.stdout+result.stderr+f'\nexit={result.returncode}\n').replace(str(Path.cwd()),'.')
    log.write(message);log.flush();print(message,flush=True)
    if result.returncode != expected or (contains and contains not in message):
        raise RuntimeError('La verificación no produjo el resultado esperado')

if __name__=='__main__':
    import _bootstrap
    import torch
    directory=Path('results/evidence');directory.mkdir(parents=True,exist_ok=True)
    with (directory/'local_verification.txt').open('w') as log:
        log.write('Verificación local '+datetime.now(timezone.utc).isoformat()+'\nDatos sintéticos; sin conclusiones de calidad ASL.\n')
        run(['-m','pytest','-q'],log)
        run(['scripts/collect_hardware.py'],log)
        run(['scripts/benchmark.py','--config','configs/smoke.yaml','--devices','cpu','--repeats','5'],log)
        if not torch.cuda.is_available():
            run(['scripts/run_cuda.py','--config','configs/smoke.yaml'],log,expected=1,contains='CUDA no disponible')
        else:
            log.write('Hay GPU disponible; ejecutar benchmark real en lugar de prueba de rechazo.\n')
    subprocess.run([sys.executable,'scripts/report_status.py'],check=True)
