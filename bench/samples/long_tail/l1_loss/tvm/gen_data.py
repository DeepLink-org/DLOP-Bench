import torch
import numpy as np

def gen_np_args(M, N):
    def gen_base(num):
        data = np.random.randn(M, N) * 100
        data = data.astype(np.float32)
        return data

    pred = gen_base(N)
    target = gen_base(N)
    return [pred, target]

def args_adaptor(np_args):
    pred = torch.from_numpy(np_args[0]).cuda()
    target = torch.from_numpy(np_args[1]).cuda()

    return [pred, target]
