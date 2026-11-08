"""Comprueba la adaptación con pesos idénticos, sin entrenar ni usar dataset."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import _bootstrap
import importlib.util
import numpy as np
import torch
import tensorflow as tf
from asl_cuda.model import BasicCNN
from asl_cuda.utils import seed_all, write_json
if __name__=='__main__':
    seed_all(42,1)
    spec=importlib.util.spec_from_file_location('original','reference/models_keras.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    tf.keras.utils.set_random_seed(42)
    keras=original.build_basic_cnn((64,64,3),29)
    model=BasicCNN().eval()
    weights=[l for l in keras.layers if l.get_weights()]
    layers=[model.features[0],model.features[3],model.hidden,model.output]
    with torch.no_grad():
        for source,destination in zip(weights,layers):
            w,b=source.get_weights()
            w=w.transpose(3,2,0,1) if w.ndim==4 else w.T
            destination.weight.copy_(torch.from_numpy(w.copy()))
            destination.bias.copy_(torch.from_numpy(b))
    x=np.random.default_rng(42).random((2,64,64,3),dtype=np.float32)
    expected=keras(x,training=False).numpy()
    with torch.no_grad(): actual=model(torch.from_numpy(x).permute(0,3,1,2)).softmax(1).numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
    assert keras.count_params()==sum(p.numel() for p in model.parameters())
    write_json('results/evidence/keras_port.json',dict(status='OK',max_abs_error=float(np.max(np.abs(actual-expected))),
                                                     parameters=keras.count_params(),rtol=1e-4,atol=1e-5))
    print('Arquitectura y pesos compartidos: equivalencia de inferencia verificada')
