# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def broadcast_tensors(tensors):
    return torch.broadcast_tensors(tensors[0])


def args_adaptor(np_args):
    data = []
    for args in np_args:
        data.append(torch.from_numpy(args).cuda())
    return [tuple(data)]


def executer_creator():
    return Executer(broadcast_tensors, args_adaptor)
