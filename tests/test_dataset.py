import csv
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from asl_cuda.dataset import ASLDataset, CLASSES

def test_real_decode_shape_range_and_label(tmp_path):
    (tmp_path/'A').mkdir()
    Image.fromarray(np.full((100,80,3),128,dtype=np.uint8)).save(tmp_path/'A'/'a.jpg')
    manifest=tmp_path/'split.csv'
    manifest.write_text('class,filename\nA,a.jpg\n')
    x,y=ASLDataset(tmp_path,manifest)[0]
    assert x.shape==(3,64,64) and x.device.type=='cpu' and y==0
    np.testing.assert_allclose(x.numpy(),128/255,atol=1e-6)
    manifest.write_text('class,filename\nA,missing.jpg\n')
    with pytest.raises(FileNotFoundError):ASLDataset(tmp_path,manifest)

def test_frozen_splits_disjoint_and_balanced():
    seen=set()
    for split,count in [('train',12180),('validation',2610),('test',2610)]:
        rows=list(csv.DictReader(Path(f'manifests/{split}_split.csv').open()))
        keys={(r['class'],r['filename']) for r in rows}
        assert len(rows)==len(keys)==count
        assert not seen & keys
        assert sorted({r['class'] for r in rows})==CLASSES
        assert all(sum(r['class']==c for r in rows)==count//29 for c in CLASSES)
        seen |= keys
