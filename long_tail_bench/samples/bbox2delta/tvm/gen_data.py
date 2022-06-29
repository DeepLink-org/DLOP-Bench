import torch
import numpy as np

def gen_np_args(N):
    def gen_base(num):
        data = np.random.randn(num, 4) * 100
        data = data.astype(np.float32)
        return data

    proposals = gen_base(N)
    gt = gen_base(N)
    return [proposals, gt]


def args_adaptor(np_args):
    proposals = torch.from_numpy(np_args[0]).cuda()
    gt = torch.from_numpy(np_args[1]).cuda()

    return [proposals, gt]
