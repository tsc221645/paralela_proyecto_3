"""Inferencia de una imagen con el checkpoint generado fuera del tiempo medido."""
import _bootstrap
import argparse
import numpy as np
from PIL import Image
import torch
from torch.nn.functional import interpolate
from asl_cuda.model import BasicCNN
from asl_cuda.utils import device_for
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();device=device_for(a.device)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=BasicCNN().to(device);model.load_state_dict(ckpt['state_dict']);model.eval()
    with Image.open(a.image) as im:x=torch.from_numpy(np.array(im.convert('RGB'),dtype=np.float32))
    x=interpolate(x.permute(2,0,1).unsqueeze(0),size=(64,64),mode='bilinear',align_corners=False,antialias=False)/255
    with torch.inference_mode():prob=model(x.to(device)).softmax(1)[0].cpu()
    i=int(prob.argmax());print({'class':ckpt['classes'][i],'probability':float(prob[i])})
