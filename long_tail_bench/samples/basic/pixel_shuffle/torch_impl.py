import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def pixel_shuffle(input_torch, upscale_factor):
    return torch.pixel_shuffle(input_torch, upscale_factor)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input_torch]


def executer_creator():
    return Executer(pixel_shuffle, args_adaptor)
