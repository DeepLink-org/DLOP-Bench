# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def pixel_shuffle(input_torch, upscale_factor):
    return torch.pixel_shuffle(input_torch, upscale_factor)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input_torch, np_args[1][0]]


def executer_creator():
    return Executer(pixel_shuffle, args_adaptor)
