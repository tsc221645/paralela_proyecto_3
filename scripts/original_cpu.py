"""Referencia Keras original conservada; comparación auxiliar, no speedup CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import _bootstrap
import argparse
import csv
import importlib.util
import json
import hashlib
from pathlib import Path
from time import perf_counter
import tensorflow as tf
from asl_cuda.utils import config, write_json, hardware
from asl_cuda.dataset import CLASSES, datasets, data_fingerprint
from asl_cuda.evidence import Monitor

def load_original():
    spec = importlib.util.spec_from_file_location('original_models', 'reference/models_keras.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def make_dataset(c, name, shuffle=False):
    rows = list(csv.DictReader((Path(c['splits_dir'])/f'{name}_split.csv').open()))
    paths = [str(Path(c['data_root'])/r['class']/r['filename']) for r in rows]
    labels = [CLASSES.index(r['class']) for r in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle: ds = ds.shuffle(len(rows), seed=c['seed'], reshuffle_each_iteration=True)
    def load(path, label):
        with tf.device('/CPU:0'):
            x = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
            return tf.cast(tf.image.resize(x, (64,64)), tf.float32)/255., label
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(c['batch_size']).prefetch(tf.data.AUTOTUNE)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/benchmark.yaml')
    a = p.parse_args(); c = config(a.config)
    if c['synthetic']: raise ValueError('Referencia exige dataset real')
    tf.config.threading.set_intra_op_parallelism_threads(c['cpu_threads'])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    original = load_original()
    out = Path(c['output'])/'original_cpu'; out.mkdir(parents=True, exist_ok=True)
    path = out/'original_cpu_runs.csv'
    if path.exists(): raise FileExistsError('Conserve resultados previos; cambie output')
    write_json(out/'metadata.json',dict(config=c, hardware=hardware(), tensorflow=tf.__version__,
               config_sha256=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest(),
               dataset_sha256=data_fingerprint(datasets(c)),
               callbacks='EarlyStopping patience=4, ReduceLROnPlateau patience=2',
               timing='fit includes validation + final test evaluate; initialization excluded'))
    for i in range(c['repeats']):
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(c['seed'])
        model = original.build_basic_cnn((64,64,3),29); original.compile_model(model,c['learning_rate'])
        # Warmup aparte; restaurar pesos y recompilar para descartar estado Adam.
        weights = model.get_weights()
        import numpy as np
        for _ in range(c['warmup_batches']):
            model.train_on_batch(np.zeros((c['batch_size'],64,64,3),dtype='float32'),np.zeros(c['batch_size'],dtype='int32'))
        model.set_weights(weights); original.compile_model(model,c['learning_rate'])
        train = make_dataset(c,'train',True); val = make_dataset(c,'validation'); test = make_dataset(c,'test')
        with Monitor(out/f'cpu_evidence_{i+1}.csv'):
            start = perf_counter()
            history=model.fit(train,validation_data=val,epochs=c['epochs'],verbose=2,
                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True),
                                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=2,min_lr=1e-6)])
            train_validation = perf_counter()-start
            t = perf_counter(); loss,accuracy=model.evaluate(test,verbose=0); evaluate=perf_counter()-t
            row=dict(run_id=i+1,variant='original_cpu',actual_epochs=len(history.epoch),train_validation_s=train_validation,
                     test_eval_s=evaluate,total_s=perf_counter()-start,loss=loss,accuracy=accuracy)
        write_json(out/f'cpu_{i+1}_epochs.json',history.history)
        with path.open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row))
            if i==0:w.writeheader()
            w.writerow(row)
        print(json.dumps(row))
