"""Informe provisional de implementación, sin inventar mediciones ASL/CUDA."""
import _bootstrap
from pathlib import Path
import os
import tempfile
import textwrap
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'asl-cuda-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
if __name__=='__main__':
    Path('report').mkdir(exist_ok=True)
    with PdfPages('report/informe_rendimiento.pdf') as pdf:
        for file in ['docs/status.md','docs/analysis_sequential.md','docs/parallel_design.md',
                     'docs/experiment_protocol.md','results/evidence/local_verification.txt','docs/contribution_log.md']:
            path=Path(file)
            if not path.exists():continue
            lines=[]
            for line in path.read_text().splitlines():
                if line.startswith('|---') or line.startswith('```'): continue
                if line.startswith('#'): line=line.lstrip('# ')
                if line.startswith('|'): line=line.strip('| ').replace(' | ', '  /  ')
                lines.extend(textwrap.wrap(line,100) or [''])
            for offset in range(0,len(lines),48):
                fig=plt.figure(figsize=(8.27,11.69))
                fig.text(.07,.96,'Informe provisional: medición ASL/CUDA pendiente',fontsize=13,va='top')
                fig.text(.07,.925,file,fontsize=9,va='top')
                fig.text(.07,.89,'\n'.join(lines[offset:offset+48]),fontsize=9,va='top',linespacing=1.45)
                fig.text(.07,.035,'Sustituir por PDF generado con la sesión real CPU/CUDA antes de entregar.',fontsize=8)
                pdf.savefig(fig);plt.close(fig)
    print('report/informe_rendimiento.pdf (PROVISIONAL)')
