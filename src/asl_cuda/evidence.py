"""Muestreo de utilización durante cada corrida; sin capturas artificiales."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import threading
import psutil

class Monitor:
    def __init__(self, path, gpu_id=0):
        self.path = Path(path)
        self.gpu_id = str(gpu_id)
        self.stop = threading.Event()
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self
    def run(self):
        process = psutil.Process()
        with self.path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['utc','pid','cpu_percent','process_cpu_percent','rss_mb','gpu_util_percent','gpu_memory_mb'])
            while not self.stop.is_set():
                util, memory = '', ''
                try:
                    r = subprocess.run(['nvidia-smi','--id='+self.gpu_id,'--query-gpu=utilization.gpu,memory.used',
                                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                    if r.returncode == 0:
                        util, memory = r.stdout.strip().split(',')
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
                w.writerow([datetime.now(timezone.utc).isoformat(), process.pid, psutil.cpu_percent(),
                            process.cpu_percent(), process.memory_info().rss/2**20, util, memory])
                f.flush()
                self.stop.wait(0.5)
    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=3)
