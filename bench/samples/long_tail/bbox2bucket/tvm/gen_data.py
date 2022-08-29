# Copyright (c) OpenComputeLab. All Rights Reserved.
import torch
import numpy as np

def gen_np_args(M):
    proposals = np.random.randn(M, 4)
    proposals = proposals.astype(np.float32)
    gt = np.random.randn(M, 4)
    gt = gt.astype(np.float32)
    return [proposals, gt]

def args_adaptor(np_args):
    proposals = torch.from_numpy(np_args[0]).cuda()
    gt = torch.from_numpy(np_args[1]).cuda()
    return [proposals, gt]
