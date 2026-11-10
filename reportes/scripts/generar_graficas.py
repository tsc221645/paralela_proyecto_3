"""Regenera las gráficas del informe detallado desde la sesión verificada."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
source_script = ROOT / "report_supplements" / "scripts" / "verificar_resultados.py"
graph_script = ROOT / "report_supplements" / "scripts" / "generar_graficas.py"
target = ROOT / "reportes" / "figuras"

if __name__ == "__main__":
    import runpy
    runpy.run_path(str(source_script), run_name="__main__")
    runpy.run_path(str(graph_script), run_name="__main__")
    target.mkdir(parents=True, exist_ok=True)
    source = ROOT / "report_supplements" / "graficas"
    for path in source.glob("*"):
        shutil.copy2(path, target / path.name)
    utilization = ROOT / "results_fast" / "raw" / "e8ad82b8761d" / "analysis" / "cpu_gpu_utilization.png"
    if utilization.exists():
        shutil.copy2(utilization, target / "utilizacion_cpu_gpu.png")
    print(f"Figuras disponibles en {target}")
