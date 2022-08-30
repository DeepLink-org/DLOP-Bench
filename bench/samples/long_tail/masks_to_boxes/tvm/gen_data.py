# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import numpy as np

def gen_np_args(M, N, K):
    def gen_base(x, y, z):
        data = np.random.randn(x, y, z) * 100
        data = data.astype(np.float32)
        return data

    boxes = gen_base(M, N, K)

    return [boxes]

def args_adaptor(np_args):
    masks = torch.from_numpy(np_args[0]).cuda()

    return [masks]
