# Copyright (c) OpenComputeLab. All Rights Reserved.

import numpy as np
import torch


def gen_np_args(M, N):
    bboxes = np.random.rand(M, N).astype(np.float32)
    gt_bboxes = np.random.rand(M, N).astype(np.float32)
    stride = np.random.rand(M, ).astype(np.float32)

    return [bboxes, gt_bboxes, stride]


def args_adaptor(np_args):
    bboxes = torch.from_numpy(np_args[0]).cuda()
    gt_bboxes = torch.from_numpy(np_args[1]).cuda()
    stride = torch.from_numpy(np_args[2]).cuda()

    return [bboxes, gt_bboxes, stride]
