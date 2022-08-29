# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer

def pad(input, pad):
    ret = torch.nn.functional.pad(input, pad)
    return ret


def args_adaptor(np_args):
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input_image, np_args[1]]


def executer_creator():
    return Executer(pad, args_adaptor)
