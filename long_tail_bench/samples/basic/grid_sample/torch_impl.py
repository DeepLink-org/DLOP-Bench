import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def grid_sample(input_torch, grid_torch, mode, padding_mode, align_coners):
    return torch.nn.functional.grid_sample(input_torch, grid_torch, mode=mode, padding_mode=padding_mode, align_corners=align_coners)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    grid_torch = torch.from_numpy(np_args[1]).to(torch.float32).cuda()

    return [input_torch, grid_torch, np_args[2], np_args[3], np_args[4]]


def executer_creator():
    return Executer(grid_sample, args_adaptor)
