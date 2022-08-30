# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import numpy as np

def gen_np_args(M, N):
    pred = np.random.rand(M, N)
    pred = pred.astype(np.float32)
    target = np.random.randint(0, 2, (M, N))
    target = target.astype(np.float32)
    return [pred, target, 2.0, 4.0]

def args_adaptor(np_args):
    pred = torch.from_numpy(np_args[0]).cuda()
    target = torch.from_numpy(np_args[1]).cuda()
    return [pred, target]
