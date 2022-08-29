# Copyright (c) OpenComputeLab. All Rights Reserved.
import numpy as np
import torch

def args_adaptor(np_args):
    boxes1 = torch.from_numpy(np_args[0]).cuda()
    boxes2 = torch.from_numpy(np_args[1]).cuda()
    return boxes1, boxes2


def gen_np_args(M1, N1, M2, N2):
    boxes1 = np.ones((M1, N1), np.float32)
    boxes2 = np.ones((M2, N2), np.float32)
    return [boxes1, boxes2]
