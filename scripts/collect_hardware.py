import _bootstrap
from asl_cuda.utils import hardware, write_json
if __name__ == '__main__':
    write_json('results/evidence/hardware.json', hardware())
    print('results/evidence/hardware.json')
