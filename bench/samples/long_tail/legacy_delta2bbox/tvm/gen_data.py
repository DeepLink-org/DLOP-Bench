# Copyright (c) OpenComputeLab. All Rights Reserved.

import numpy as np
import torch

def gen_np_args(M, N):
    def gen_base(row, column):
        data = np.random.rand(row, column)
        data = data.astype(np.float32)
        return data

    proposals = gen_base(M, N)
    gt = gen_base(M, N)
    return [proposals, gt]

def args_adaptor(np_args):
    proposals = torch.from_numpy(np_args[0]).cuda()
    gt = torch.from_numpy(np_args[1]).cuda()

    return [proposals, gt]
