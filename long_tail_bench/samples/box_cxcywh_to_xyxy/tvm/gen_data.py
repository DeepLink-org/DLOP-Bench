import torch
import numpy as np

def gen_np_args(M, N):
    x = np.random.randn(M, N)
    x = x.astype(np.float32)
    return [x]

def args_adaptor(np_args):
    x = torch.from_numpy(np_args[0]).cuda()
    return [x]
