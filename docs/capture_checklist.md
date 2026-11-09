# Checklist de capturas para la entrega

Durante la corrida final:

1. En CPU, abrir Administrador de tareas → Rendimiento → CPU y mostrar el proceso Python, uso de CPU y memoria. Capturar también la ventana de la terminal con el comando y el backend CPU.
2. En CUDA, abrir una terminal separada y ejecutar `nvidia-smi -l 1`. Capturar nombre de GPU, memoria usada, utilización GPU y el proceso Python.
3. Tomar una captura de la terminal donde aparezca `model_device: cuda:0` o una fila `variant=cuda` del CSV.
4. Guardar las imágenes con nombres `cpu_execution.png`, `cuda_execution.png` y `cuda_nvidia_smi.png` dentro de la carpeta de entrega.
5. No capturar credenciales ni rutas personales innecesarias.

El monitor automático también deja CSV en `results_fast/evidence/<ID>`, pero las capturas manuales son necesarias para la evidencia visual solicitada por la guía.
