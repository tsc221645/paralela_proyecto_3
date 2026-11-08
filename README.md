# Proyecto 3: ASL en CPU y CUDA

Refactorización del Laboratorio 3 para comparar su CNN básica en PyTorch CPU/CUDA. Preparación de imágenes en CPU, entrenamiento CUDA explícito, cinco repeticiones, tiempos por fase, huellas de datos/código, evidencia de uso y generación de gráficas/PDF.

El original y las especificaciones están excluidos en `.gitignore` según lo solicitado. Las arquitecturas originales y hashes de procedencia están en `reference/`; los splits portables en `manifests/`. No hay dataset de entrenamiento incluido. Los resultados sintéticos solo verifican el programa.

## Instalación en otra computadora

Recomendado: Python 3.12, Linux o Windows con GPU NVIDIA y controlador compatible. En Windows usar terminal con el entorno activado; no requiere WSL para PyTorch. Ejecutar siempre desde esta raíz.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

El wheel de torch debe tener CUDA compatible con el controlador. Si la comprobación devuelve False, instalar el wheel indicado por el [selector oficial PyTorch](https://pytorch.org/get-started/locally/) para el sistema, manteniendo torch 2.9.1 para este entorno fijado. No basta tener el toolkit instalado. Para una instalación exclusivamente CPU puede usarse el índice `https://download.pytorch.org/whl/cpu`. `requirements.txt` fija dependencias directas; cada ejecución registra versiones efectivas, no constituye un lock transitivo multiplataforma.

Descargar el conjunto **de entrenamiento** [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), extraerlo y ajustar `data_root` en `configs/benchmark.yaml` a la carpeta que contiene A/, B/, ... space/. No entrenar con el ZIP pequeño de test del laboratorio. No es necesario copiar `lab3_datascience` a la computadora GPU.

```bash
python scripts/prepare_data.py
python scripts/collect_hardware.py
python -m pytest -q
python scripts/benchmark.py --config configs/benchmark.yaml --repeats 5
```

Se ejecutan CPU y GPU **en esa misma computadora**. CUDA solicitada sin GPU produce error, sin reemplazarla por CPU. Conservar la carpeta completa `results/raw/ID` y `results/evidence/ID`. La carpeta `analysis/` de la sesión contiene tabla, gráficas y `informe_rendimiento.pdf`; copiar el PDF real final a `report/` al cerrar la entrega. Tomar además capturas manuales con CPU/GPU durante ejecución. Ver [protocolo](docs/experiment_protocol.md).

Para un solo backend: `python scripts/run_cpu.py --config configs/benchmark.yaml` o `python scripts/run_cuda.py --config configs/benchmark.yaml`. Para volver a generar informe: `python scripts/plot_results.py --input results/raw/ID`.

Referencia Keras auxiliar, opcional para correr el benchmark principal pero necesaria para cerrar su comparación histórica:

```bash
python -m pip install -r requirements-original.txt
python scripts/original_cpu.py --config configs/benchmark.yaml
python scripts/check_keras_port.py
```

Se mantienen callbacks originales en esa referencia. Los experimentos principales fijan épocas/LR para igualar trabajo; no atribuir diferencias Keras/PyTorch a CUDA. La auditoría explica el alcance y las limitaciones.

## Verificación sin GPU ni datos

```bash
python -m pytest -q
python scripts/benchmark.py --config configs/smoke.yaml --devices cpu --repeats 5
```

Esto prueba entrenamiento, repetibilidad, CSV y PDF con datos sintéticos. La prueba numérica CUDA se omite explícitamente si falta GPU. El [PDF provisional](report/informe_rendimiento.pdf) documenta la implementación y pendientes; no contiene resultados ASL/CUDA inventados. Se regenera con `python scripts/report_status.py`. Ver [estado de fases](docs/status.md), [auditoría](docs/analysis_sequential.md), [diseño](docs/parallel_design.md) y [aportes pendientes](docs/contribution_log.md).

El último modelo de cada backend se guarda como `cpu_model.pt`/`cuda_model.pt` en la sesión, fuera del tiempo medido. Para inferencia: `python scripts/predict.py --checkpoint results/raw/ID/cuda_model.pt --image mano.jpg --device cuda`. Los checkpoints pesados se excluyen de Git; conservarlos junto con los resultados de la entrega.
