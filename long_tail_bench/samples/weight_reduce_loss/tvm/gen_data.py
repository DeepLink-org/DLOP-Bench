import torch
import numpy as np

def gen_np_args(M, N):
    loss_in = np.random.randn(M, N).astype(np.float32)
    weight_in = np.random.randn(M, N).astype(np.float32)
    return [loss_in, weight_in]

def args_adaptor(np_args):
    loss_in = torch.from_numpy(np_args[0]).cuda()
    weight_in = torch.from_numpy(np_args[1]).cuda()

    return [loss_in, weight_in]
