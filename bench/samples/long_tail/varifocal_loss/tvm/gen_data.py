# Copyright (c) OpenComputeLab. All Rights Reserved.
import torch
import numpy as np

def gen_np_args(M, N):
    pred = np.random.randn(M, N).astype(np.float32)
    target = np.random.randn(M, N).astype(np.float32)
    return [pred, target]

def args_adaptor(np_args):
    pred = torch.from_numpy(np_args[0]).cuda()
    target = torch.from_numpy(np_args[1]).cuda()
    # pred.requires_grad = True

    return [pred, target]
