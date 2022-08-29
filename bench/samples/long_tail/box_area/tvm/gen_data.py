# Copyright (c) OpenComputeLab. All Rights Reserved.
import numpy as np
import torch


def gen_np_args(M, N):
    boxes = np.random.randn(M, N)
    boxes = boxes.astype(np.float32)
    return [boxes]


def args_adaptor(np_args):
    boxes = torch.from_numpy(np_args[0]).cuda()
    return boxes
