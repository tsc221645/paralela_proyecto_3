# Análisis de la solución secuencial

## Flujo de datos

| Operación | Descripción |
|---|---|
| Enumeración | Los manifests congelados identifican las imágenes y su clase. |
| Lectura | `PIL.Image` abre cada JPEG y lo convierte a RGB en CPU. |
| Resize | PyTorch `interpolate` redimensiona cada imagen a 64x64 en CPU. |
| Normalización | Los valores float32 se dividen entre 255. |
| Formación de lotes | `DataLoader` construye lotes de 64 muestras. |
| Transferencia | En CUDA, los tensores de entrada y etiquetas pasan de CPU a GPU. |
| CNN | Convoluciones, activaciones, pooling, Dense, loss y Adam se ejecutan en el dispositivo seleccionado. |
| Evaluación | Se calculan loss y accuracy sobre validation y test. |
| Persistencia | Se guardan CSV, metadata, historial por época y el último checkpoint. |

## Operaciones repetitivas

| Operación | Datos | Cantidad en la corrida rápida |
|---|---|---:|
| Lectura/preprocesamiento | 12,180 imágenes train por época | 5 épocas x 12,180 |
| Forward/backward/Adam | 191 lotes train por época | 5 x 191 |
| Validación | 2,610 imágenes por época | 5 x 2,610 |
| Test final | 2,610 imágenes | 1 x 2,610 |
| Repeticiones | Configuración completa | 3 o 5 corridas por backend |

El pipeline conserva el I/O, la decodificación JPEG, resize y normalización en CPU. El modelo y la matemática de entrenamiento se colocan explícitamente en `cuda:0` cuando se ejecuta el backend CUDA; no existe fallback silencioso a CPU.
