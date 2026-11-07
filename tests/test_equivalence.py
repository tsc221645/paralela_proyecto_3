import copy
import pytest
import torch
from asl_cuda.model import BasicCNN
from asl_cuda.utils import device_for, seed_all

def test_model_parameter_count_and_learning():
    seed_all(42,1)
    model=BasicCNN()
    assert sum(p.numel() for p in model.parameters())==2120413
    model.eval()
    x=torch.rand(2,3,64,64); y=torch.tensor([0,1])
    opt=torch.optim.SGD(model.parameters(),lr=.01)
    before=torch.nn.functional.cross_entropy(model(x),y)
    opt.zero_grad();before.backward();opt.step()
    after=torch.nn.functional.cross_entropy(model(x),y)
    assert torch.isfinite(after) and after<before
    assert model(x).shape==(2,29)

def test_cuda_failure_explicit():
    if torch.cuda.is_available():pytest.skip('CUDA sí disponible')
    with pytest.raises(RuntimeError,match='CUDA no disponible'):device_for('cuda')

@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requiere GPU NVIDIA CUDA real')
def test_cpu_cuda_forward_backward_and_update():
    seed_all(42,1)
    cpu=BasicCNN().eval();gpu=copy.deepcopy(cpu).cuda().eval()
    x=torch.rand(2,3,64,64);y=torch.tensor([0,28])
    a=cpu(x);b=gpu(x.cuda())
    torch.testing.assert_close(a,b.cpu(),rtol=1e-4,atol=1e-5)
    la=torch.nn.functional.cross_entropy(a,y);lb=torch.nn.functional.cross_entropy(b,y.cuda())
    la.backward();lb.backward()
    for p,q in zip(cpu.parameters(),gpu.parameters()):
        torch.testing.assert_close(p.grad,q.grad.cpu(),rtol=1e-3,atol=1e-5)
    # SGD evita amplificar gradientes cercanos a cero como puede hacer Adam.
    torch.optim.SGD(cpu.parameters(),lr=.001).step();torch.optim.SGD(gpu.parameters(),lr=.001).step()
    for p,q in zip(cpu.parameters(),gpu.parameters()):
        torch.testing.assert_close(p,q.cpu(),rtol=1e-4,atol=1e-5)
