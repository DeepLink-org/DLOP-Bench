# Copyright (c) OpenComputeLab. All Rights Reserved.
import torch
import numpy as np

def gen_np_args(M, N):
    proposals = np.random.rand(M, N)
    proposals = proposals.astype(np.float32)
    return [proposals]

def args_adaptor(np_args):
    proposals = torch.from_numpy(np_args[0]).cuda()
    return [proposals]
