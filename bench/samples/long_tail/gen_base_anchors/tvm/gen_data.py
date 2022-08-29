# Copyright (c) OpenComputeLab. All Rights Reserved.
import numpy as np
import torch


def gen_np_args(M, N, K):
    ratios = np.random.randn(N)
    ratios = ratios.astype(np.float32)
    scales = np.random.randn(K)
    scales = scales.astype(np.float32)
    return [ratios, scales]

def args_adaptor(np_args):
    ratios = torch.from_numpy(np_args[0]).cuda()
    scales = torch.from_numpy(np_args[1]).cuda()
    return [ratios, scales]
