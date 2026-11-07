import _bootstrap
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from asl_cuda.reporting import aggregate, write_csv, PHASES

def page(pdf, title, paragraphs):
    import textwrap
    fig=plt.figure(figsize=(8.27,11.69))
    fig.text(.08,.95,title,fontsize=16,va='top')
    y=.90
    for paragraph in paragraphs:
        text='\n'.join(textwrap.wrap(paragraph,95))
        fig.text(.08,y,text,fontsize=10,va='top')
        y-=.022*(text.count('\n')+1)+.018
    pdf.savefig(fig);plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,help='Carpeta de UNA sesión results/raw/ID')
    p.add_argument('--output')
    a=p.parse_args(); root=Path(a.input)
    groups,meta,summary,speedups=aggregate(root)
    out=Path(a.output) if a.output else root/'analysis';out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'summary.csv',summary);write_csv(out/'speedup.csv',speedups)
    synthetic=next(iter(meta.values()))['config']['synthetic']
    label='SINTÉTICO: prueba de ejecución, sin validez ASL' if synthetic else 'ASL Alphabet'
    fig,ax=plt.subplots(figsize=(10,5))
    devices=list(groups); width=.8/len(devices)
    for j,device in enumerate(devices):
        rows=[r for r in summary if r['device']==device and r['phase'] in PHASES]
        ax.bar([i+j*width for i in range(len(rows))],[r['mean'] for r in rows],width,
               yerr=[r['std'] for r in rows],label=device,capsize=3)
    ax.set_xticks(range(len(PHASES)),PHASES);ax.set_ylabel('Segundos (media ± desviación estándar)')
    ax.set_title(label);ax.legend();fig.tight_layout();fig.savefig(out/'times_by_phase.png');plt.close(fig)
    if speedups:
        fig,ax=plt.subplots(figsize=(8,4));ax.bar([r['phase'] for r in speedups],[r['speedup'] for r in speedups])
        ax.axhline(1,color='black');ax.set_ylabel('CPU / CUDA');ax.set_title(label)
        fig.tight_layout();fig.savefig(out/'speedup_by_phase.png');plt.close(fig)
    evidence=root.parent.parent/'evidence'/root.name
    files=list(evidence.glob('*.csv'))
    if files:
        fig,ax=plt.subplots(figsize=(10,4))
        for path in files:
            import csv
            rows=list(csv.DictReader(path.open()))
            ax.plot([i*.5 for i in range(len(rows))],[float(r['cpu_percent']) for r in rows],label=path.stem+' CPU')
            if rows and all(r['gpu_util_percent'].strip() for r in rows):
                ax.plot([i*.5 for i in range(len(rows))],[float(r['gpu_util_percent']) for r in rows],label=path.stem+' GPU')
        ax.set_xlabel('Índice de muestra × 0.5 s (aproximado)');ax.set_ylabel('Uso global (%)')
        ax.legend(fontsize=6);fig.tight_layout();fig.savefig(out/'cpu_gpu_utilization.png');plt.close(fig)
    with PdfPages(out/'informe_rendimiento.pdf') as pdf:
        page(pdf,'Proyecto 3: CPU y CUDA para ASL',[
            label,
            'Objetivo: separar lectura y preparación CPU del entrenamiento neuronal CUDA, conservando arquitectura, particiones e hiperparámetros.',
            'Caso real: Laboratorio 3, CNN básica Keras adaptada a PyTorch: Conv 3x3 32 + ReLU + MaxPool; Conv 3x3 64 + ReLU + MaxPool; Flatten NHWC; Dense 128 + ReLU; Dropout 0.3; salida de 29 clases. 2,120,413 parámetros. CrossEntropyLoss incorpora log-softmax.',
            'Datos: JPEG RGB 64x64; normalización /255; resize bilineal CPU sin antialias. Particiones congeladas 12180/2610/2610. Las diferencias de decodificación JPEG entre frameworks impiden prometer igualdad bit a bit con el notebook.',
            'Inventario: enumeración, lectura, JPEG, resize, normalización y lotes en CPU; convoluciones, productos matriciales, activaciones, loss, gradientes y Adam en CUDA. Copias H2D una por tensor de entrada y D2H de dos métricas por evaluación.',
            'Frecuencias: una decodificación por acceso; ceil(Ntrain/batch) forward/backward/Adam por época; ceil(Nval/batch) forward por época; ceil(Ntest/batch) al final. Sin caché persistente.',
            'Diseño: muestras -> lotes -> tensores -> kernels CUDA/cuDNN. Hilos, bloques y grids gestionados por PyTorch; no se afirma haber implementado kernels propios.',
        ])
        m=next(iter(meta.values()))
        page(pdf,'Ambiente y protocolo',[
            json.dumps(m['hardware'],ensure_ascii=False).split('"nvidia_smi"')[0][:1500],
            'Configuración: '+json.dumps(m['config'],ensure_ascii=False),
            'SHA256 datos: '+m['dataset_sha256'],
            'Cada repetición reinicia pesos, semillas y Adam; warmup fuera del tiempo. CPU usa un hilo y num_workers=0 en comparación principal. FP32, TF32 desactivado. GPU requiere CUDA explícito; no hay fallback.',
            'Preparación = espera de next(loader), incluida decodificación cuando workers=0. Train = forward/loss/backward/Adam; eval = forward/loss/agregación de validación y test. H2D/D2H separados. Total incluye construcción de modelo y loaders y fases de todas las épocas; excluye hash previo, importación y warmup.',
            'Medición perf_counter con sincronización antes/después de segmentos CUDA; CUDA Events miden el segmento de entrenamiento por separado. Las sincronizaciones reducen solapamiento: esta instrumentación diagnostica fases, no representa el máximo throughput posible.',
            'Resumen de resultados en la página siguiente. La desviación es muestral. Speedup = media CPU / media CUDA; reducción = 100*(1-media CUDA/media CPU). H2D/D2H se reportan como costos absolutos.',
        ])
        lines=[f'{r["device"]} {r["phase"]}: media={r["mean"]:.6f}, std={r["std"]:.6f}, mediana={r["median"]:.6f}, n={r["n"]}' for r in summary]
        page(pdf,'Resultados medidos',lines[:20])
        page(pdf,'Discusión y limitaciones',[
            *[f'{r["phase"]}: speedup {r["speedup"]:.3f}; reducción {r["reduction_percent"]:.2f}%' for r in speedups],
            'No hay comparación CUDA medida en esta sesión.' if not speedups else 'Un speedup menor que uno indica degradación; se conserva sin ocultar costos.',
            'Las imágenes permanecen en CPU por I/O, decodificación y requisito del proyecto. Las convoluciones y álgebra densa ofrecen paralelismo por lote, canales y píxeles que CUDA puede explotar.',
            'Ayudan lotes suficientes y buena alimentación de datos; limitan CPU/I/O, transferencias, VRAM, lanzamientos y carga pequeña. Pin memory y workers son experimentos separados; no se presume que mejoren.',
            'Evidencia: metadata JSON y CSV de utilización por corrida. Muestreo global de GPU, no atribución exclusiva al proceso; para entrega adjuntar captura de nvidia-smi durante la ejecución. La memoria reportada por PyTorch es propia del proceso.',
            'Comparación con Keras original: script auxiliar original_cpu.py; no atribuir diferencias de frameworks, JPEG, inicialización ni callbacks a CUDA. El notebook completo no se ejecutó aquí por ausencia del dataset. El experimento principal compara únicamente PyTorch CPU/CUDA.',
            'Métricas finales: test separado, validación cada época. Datos sintéticos solo comprueban funcionamiento; no prueban calidad ASL. Validación de equivalencia y tolerancias en tests/test_equivalence.py.',
            'Equipo, aportes y URL de repositorio: deben completarse por los tres integrantes en docs/contribution_log.md. No se inventan autores, commits, revisiones ni ejecuciones GPU.',
        ])
        for name in ('times_by_phase.png','speedup_by_phase.png','cpu_gpu_utilization.png'):
            if (out/name).exists():
                fig,ax=plt.subplots(figsize=(11,7));ax.imshow(plt.imread(out/name));ax.axis('off');pdf.savefig(fig);plt.close(fig)
    print(out/'informe_rendimiento.pdf')
if __name__=='__main__':main()
