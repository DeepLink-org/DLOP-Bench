# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn.functional as F
import numpy as np
from bench.core.executer import Executer



def interpolateBP(input_torch, output_size, scale_factor, mode, align_corners, recompute_scale_factor):
    ret = F.interpolate(input_torch, output_size, scale_factor, mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    ret.backward(torch.ones_like(ret))
    return ret


def args_adaptor(np_args):
    input_torch = torch.tensor(np_args[0], requires_grad=True).to(torch.float32).cuda()
    output_size = np_args[1]
    scale_factor = np_args[2]
    mode = np_args[3]
    align_corners = np_args[4]
    recompute_scale_factor = np_args[5]

    return [input_torch, output_size, scale_factor, mode, align_corners, recompute_scale_factor]


def executer_creator():
    return Executer(interpolateBP, args_adaptor)
