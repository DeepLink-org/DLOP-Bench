# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import numpy as np

def gen_np_args(N):
    priors = np.random.randn(N, 4).astype(np.float32)
    tblr = np.random.randn(N, 4).astype(np.float32)
    return [priors, tblr]

def args_adaptor(np_args):
    priors = torch.from_numpy(np_args[0]).cuda()
    tblr = torch.from_numpy(np_args[1]).cuda()

    return [priors, tblr]
