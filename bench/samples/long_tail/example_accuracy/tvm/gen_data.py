# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import numpy as np

def gen_np_args(M, N):
    output = np.random.randn(M, N)
    output = output.astype(np.float32)
    target = np.random.randint(0, 1000, (M, ))
    target = target.astype(np.int64)
    return [output, target]

def args_adaptor(np_args):
    output = torch.from_numpy(np_args[0]).cuda()
    target = torch.from_numpy(np_args[1]).cuda()
    return [output, target]
