# Copyright(c) OpenMMLab. All Rights Reserved.
# Copied from
import numpy as np
import torch


def gen_np_args(N, M):
    output = np.random.rand(N, M).astype(np.float32)
    target = np.random.randint(0, 1000, size=(N, ), dtype=np.int64)
    return [output, target]

def args_adaptor(np_args):
    output = torch.from_numpy(np_args[0]).cuda()
    target = torch.from_numpy(np_args[1]).cuda()

    return [output, target]

