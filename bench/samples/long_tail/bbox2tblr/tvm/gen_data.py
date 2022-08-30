# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import numpy as np


def gen_np_args(M, N):
    priors = np.random.randn(M, N)
    priors = priors.astype(np.float32)
    gts = np.random.randn(M, N)
    gts = gts.astype(np.float32)
    return [priors, gts]


def args_adaptor(np_args):
    priors = torch.from_numpy(np_args[0]).cuda()
    gts = torch.from_numpy(np_args[1]).cuda()
    return [priors, gts]
