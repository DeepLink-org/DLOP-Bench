# Copyright (c) OpenComputeLab. All Rights Reserved.
import numpy as np
import torch

def gen_np_args(M, N):
    def gen_base(row, column):
        data = np.zeros((row, column))
        data = data.astype(np.float32)
        data[:, 3] = 1
        return data

    rois = gen_base(M, N)

    return [rois]


def args_adaptor(np_args):
    rois = torch.from_numpy(np_args[0]).cuda()
    return [rois]
