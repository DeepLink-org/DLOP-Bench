import torch
import torch.nn.functional as F
import numpy as np
from long_tail_bench.core.executer import Executer



def interpolate(input_torch, output_size, scale_factor, mode, align_corners, recompute_scale_factor):

    return F.interpolate(input_torch, output_size, scale_factor, mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    output_size = np_args[1]
    scale_factor = np_args[2]
    mode = np_args[3]
    align_corners = np_args[4]
    recompute_scale_factor = np_args[5]

    return [input_torch, output_size, scale_factor, mode, align_corners, recompute_scale_factor]


def executer_creator():
    return Executer(interpolate, args_adaptor)
