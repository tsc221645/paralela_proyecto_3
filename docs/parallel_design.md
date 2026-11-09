# Diseño de paralelización CUDA

## Partición del trabajo

El lote es la unidad de trabajo. PyTorch/cuDNN distribuye automáticamente los elementos del lote, canales y posiciones espaciales de las convoluciones entre grids, bloques e hilos CUDA. Los productos matriciales de las capas Dense también se ejecutan mediante kernels paralelos de la biblioteca.

| Fase | CPU | GPU CUDA |
|---|---|---|
| Archivos y JPEG | Sí | No |
| Resize y normalización | Sí | No |
| Copia de lote | Origen | Destino H2D |
| Conv2D/ReLU/MaxPool | No | Sí |
| Dense, loss, gradientes y Adam | No | Sí |
| Métricas finales | Recibe dos escalares | Calcula acumuladores |

La implementación utiliza CUDA mediante PyTorch (`tensor.to('cuda:0')`, cuDNN y kernels de PyTorch). No se afirma que existan kernels escritos manualmente con Numba o CUDA C++; si la rúbrica exige kernels propios, esa sería una extensión adicional.

## Medición

Cada segmento CUDA se sincroniza antes y después de medirlo. El entrenamiento GPU además registra CUDA Events. H2D y D2H se reportan por separado; `total_s` se mide independientemente y no se obtiene sumando fases superpuestas.

La comparación CPU/CUDA se ejecuta secuencialmente en la misma máquina. Ejecutarlos simultáneamente contaminaría el speedup por competencia por CPU, disco, memoria y GPU.
