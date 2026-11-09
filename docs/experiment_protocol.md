# Protocolo de experimento

## Datos y configuración

- Dataset: ASL Alphabet, 29 clases, 87,000 imágenes.
- Particiones congeladas: train 12,180; validation 2,610; test 2,610.
- Resolución: 64x64 RGB, float32, normalización `/255`.
- Modelo: CNN básica de 2,120,413 parámetros.
- Comparación: misma semilla, manifests, hiperparámetros y máquina.
- Preparación CPU; entrenamiento CPU o CUDA según backend.

## Corrida final recomendada

```powershell
conda activate dm1
python scripts/benchmark.py --config configs/benchmark_fast.yaml --devices cpu cuda --repeats 5
```

La configuración rápida usa 5 épocas, batch 64 y 16 hilos CPU. Las cinco repeticiones cumplen el requisito de obtener medias y desviaciones más estables sin dejar la configuración oficial de 20 épocas ejecutándose durante horas.

## Métricas

`speedup = media_cpu / media_cuda`

`reducción = 100 * (1 - media_cuda / media_cpu)`

Se conservan metadata JSON, CSV por backend, muestras de utilización, historial por época, checkpoints y PDF en `results_fast/raw/<ID>`.
