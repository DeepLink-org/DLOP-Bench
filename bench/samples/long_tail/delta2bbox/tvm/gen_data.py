# Copyright (c) OpenComputeLab. All Rights Reserved.

import numpy as np
import torch


def gen_np_args(M, N):
    rois = np.random.randn(M, N)
    deltas = np.random.randn(M, N)

    return [rois, deltas]


def args_adaptor(np_args):
    rois = torch.from_numpy(np_args[0]).cuda()
    deltas = torch.from_numpy(np_args[1]).cuda()

    return [rois, deltas]
